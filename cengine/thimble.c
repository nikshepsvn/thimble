/* thimble.c — single-file C inference engine for Thimble.
 *
 * A line-faithful port of the Python inference stack (model.py, tokenizer.py,
 * grammar.py, retrieve.py, render.py): same tokenizer contract, same five
 * decision points, same gating constants. Decisions are argmaxes over
 * well-separated scores, so fp32-vs-fp32 float noise between this and PyTorch
 * almost never flips one; parity is verified against demo.py output.
 *
 *   make && ./thimble -w thimble.bin -t tokenizer.bin -c demo_catalog.json \
 *       "make a reservation at Nobu for 2 people at 7pm"
 *
 * Output is dumps_calls-compatible compact JSON on stdout, one line per query.
 * With --jsonl FILE it instead reads {"query":...,"tools":[...]} rows and
 * prints one prediction line per row (for parity/eval against the Python
 * harness).
 *
 * Not ported (disabled by default in Python, measured worse there):
 * pointer_copy, copy_span_value, gen_templated. temp>0 sampling is not ported
 * either; every choice is the temp=0 argmax.
 */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

static long g_forwards = 0;   /* one per token fed through the trunk */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ util */

static void die(const char *msg) {
    fprintf(stderr, "thimble: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz);
    if (!p) die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    char *p = xmalloc(strlen(s) + 1);
    strcpy(p, s);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

/* --------------------------------------------------------------- vec ops */

static void matvec(const float *w, const float *x, float *y, int out, int in) {
#ifdef __APPLE__
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out, in, 1.0f, w, in, x, 1, 0.0f, y, 1);
#else
    for (int o = 0; o < out; o++) {
        const float *row = w + (size_t)o * in;
        float acc = 0.0f;
        for (int i = 0; i < in; i++) acc += row[i] * x[i];
        y[o] = acc;
    }
#endif
}

/* Y (n x out) = X (n x in) @ W^T, W stored (out x in) row-major (torch Linear) */
static void matmul_nt(const float *w, const float *x, float *y, int n, int out, int in) {
    if (n == 1) {
        matvec(w, x, y, out, in);
        return;
    }
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, out, in,
                1.0f, x, in, w, in, 0.0f, y, out);
#else
    for (int r = 0; r < n; r++) matvec(w, x + (size_t)r * in, y + (size_t)r * out, out, in);
#endif
}

static void rmsnorm(const float *x, const float *w, float *y, int d) {
    /* matches model.py RMSNorm: fp32 mean-of-squares, eps 1e-6 */
    double ss = 0.0;
    for (int i = 0; i < d; i++) ss += (double)x[i] * x[i];
    float rms = (float)(1.0 / sqrt(ss / d + 1e-6));
    for (int i = 0; i < d; i++) y[i] = x[i] * rms * w[i];
}

static void softmax_(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] = (float)(x[i] / sum);
}

static int argmax_f(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

/* log_softmax normalizer of a logit vector (fp32, like Python .float()) */
static double logsumexp(const float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += exp((double)x[i] - mx);
    return mx + log(sum);
}

/* ---------------------------------------------------------------- model */

/* A matmul weight: fp32 rows, or per-row-scaled int8 (v2 files). */
typedef struct {
    const float *f;       /* fp32 rows, or NULL when quantized */
    const int8_t *q;      /* int8 rows */
    const float *sc;      /* per-row scales */
    int out, in;
} MW;

typedef struct {
    const float *n1, *qn, *kn, *n2, *n3, *n4;
    MW q, k, v, o, gate, w1, w2, w3;
} Layer;

typedef struct {
    int vocab, d, layers, heads, kv, ffn, max_seq, quant;
    float theta;
    int hd, kvd;      /* head_dim, kv*head_dim */
    MW embed, name_head;
    const float *norm;
    Layer *L;
    char *blob;       /* whole file */
    float *deq;       /* scratch for dequant-then-sgemm on chunks */
} Model;

typedef struct { const char *p; } Cursor;

static const float *take(Cursor *c, size_t n) {
    const float *r = (const float *)c->p;
    c->p += n * 4;
    return r;
}

static MW take_mw(Cursor *c, int out, int in, int quant) {
    MW w;
    w.out = out;
    w.in = in;
    if (quant) {
        w.f = NULL;
        w.sc = (const float *)c->p;
        c->p += (size_t)out * 4;
        w.q = (const int8_t *)c->p;
        c->p += (size_t)out * in;
    } else {
        w.f = (const float *)c->p;
        c->p += (size_t)out * in * 4;
        w.q = NULL;
        w.sc = NULL;
    }
    return w;
}

/* fetch one weight row as fp32 into dst (embedding lookup) */
static void mw_row(const MW *w, int row, float *dst) {
    if (w->f) {
        memcpy(dst, w->f + (size_t)row * w->in, sizeof(float) * w->in);
    } else {
        const int8_t *r = w->q + (size_t)row * w->in;
        float s = w->sc[row];
        for (int i = 0; i < w->in; i++) dst[i] = r[i] * s;
    }
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define HAVE_SDOT 1
#endif

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define HAVE_WASM_SIMD 1
#endif

/* y = W x (single row of activations).
 *
 * Quantized fast path: quantize x once to int8 (absmax), then each output row
 * is an int8·int8 dot — with sdot on ARM that is 16 MACs per instruction and
 * the whole sweep touches 1/4 the bytes of fp32. The activation quantization
 * is the one numeric liberty the engine takes; it is measured (row agreement
 * and eval score vs fp32), not assumed. */
static void mw_matvec(const MW *w, const float *x, float *y) {
    if (w->f) {
        matvec(w->f, x, y, w->out, w->in);
        return;
    }
    int in = w->in;
    static int8_t *xq = NULL;
    static int xq_cap = 0;
    if (in > xq_cap) {
        free(xq);
        xq = xmalloc((size_t)in + 16);
        xq_cap = in;
    }
    float amax = 1e-12f;
    for (int i = 0; i < in; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    float sx = amax / 127.0f, inv = 127.0f / amax;
    for (int i = 0; i < in; i++) xq[i] = (int8_t)lrintf(x[i] * inv);
#ifdef HAVE_SDOT
    for (int o = 0; o < w->out; o++) {
        const int8_t *row = w->q + (size_t)o * in;
        int32x4_t acc = vdupq_n_s32(0);
        int i = 0;
        for (; i + 16 <= in; i += 16)
            acc = vdotq_s32(acc, vld1q_s8(row + i), vld1q_s8(xq + i));
        int32_t s = vaddvq_s32(acc);
        for (; i < in; i++) s += row[i] * xq[i];
        y[o] = w->sc[o] * sx * (float)s;
    }
#elif defined(HAVE_WASM_SIMD)
    for (int o = 0; o < w->out; o++) {
        const int8_t *row = w->q + (size_t)o * in;
        v128_t acc = wasm_i32x4_splat(0);
        int i = 0;
        for (; i + 16 <= in; i += 16) {
            v128_t a = wasm_v128_load(row + i);
            v128_t b = wasm_v128_load(xq + i);
            /* i8 x i8 -> i16 pairs -> pairwise i32 accumulate; products are
             * bounded by 127*127 so the i16 lanes cannot overflow */
            v128_t lo = wasm_i16x8_extmul_low_i8x16(a, b);
            v128_t hi = wasm_i16x8_extmul_high_i8x16(a, b);
            acc = wasm_i32x4_add(acc, wasm_i32x4_extadd_pairwise_i16x8(lo));
            acc = wasm_i32x4_add(acc, wasm_i32x4_extadd_pairwise_i16x8(hi));
        }
        int32_t s = wasm_i32x4_extract_lane(acc, 0) + wasm_i32x4_extract_lane(acc, 1)
                  + wasm_i32x4_extract_lane(acc, 2) + wasm_i32x4_extract_lane(acc, 3);
        for (; i < in; i++) s += row[i] * xq[i];
        y[o] = w->sc[o] * sx * (float)s;
    }
#else
    for (int o = 0; o < w->out; o++) {
        const int8_t *row = w->q + (size_t)o * in;
        int32_t s = 0;
        for (int i = 0; i < in; i++) s += row[i] * xq[i];
        y[o] = w->sc[o] * sx * (float)s;
    }
#endif
}

/* Y (n x out) = X (n x in) @ W^T; quantized weights are dequantized into the
 * model's scratch once per call, so chunk prefill keeps sgemm speed. */
static void mw_matmul(Model *m, const MW *w, const float *x, float *y, int n) {
    if (n == 1) {
        mw_matvec(w, x, y);
        return;
    }
    const float *wf = w->f;
    if (!wf) {
#ifndef __APPLE__
        /* no BLAS to feed: the vectorized int8 matvec per row beats
         * dequant + scalar fp32 loops on both memory traffic and compute */
        for (int r = 0; r < n; r++)
            mw_matvec(w, x + (size_t)r * w->in, y + (size_t)r * w->out);
        return;
#else
        for (int o = 0; o < w->out; o++) {
            const int8_t *row = w->q + (size_t)o * w->in;
            float *dst = m->deq + (size_t)o * w->in;
            float s = w->sc[o];
            for (int i = 0; i < w->in; i++) dst[i] = row[i] * s;
        }
        wf = m->deq;
#endif
    }
    matmul_nt(wf, x, y, n, w->out, w->in);
}

static Model *model_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open weights file");
    uint32_t magic = 0, ver = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        magic != 0x424D4854u || (ver != 1 && ver != 2))
        die("bad weights header");
    int32_t hdr[7];
    float theta = 0.0f;
    if (fread(hdr, 4, 7, f) != 7 || fread(&theta, 4, 1, f) != 1)
        die("bad weights header");
    Model *m = xcalloc(1, sizeof(Model));
    m->vocab = hdr[0]; m->d = hdr[1]; m->layers = hdr[2]; m->heads = hdr[3];
    m->kv = hdr[4]; m->ffn = hdr[5]; m->max_seq = hdr[6]; m->theta = theta;
    m->quant = (ver == 2);
    m->hd = m->d / m->heads;
    m->kvd = m->kv * m->hd;
    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    size_t nbytes = (size_t)(ftell(f) - pos);
    fseek(f, pos, SEEK_SET);
    m->blob = xmalloc(nbytes);
    if (fread(m->blob, 1, nbytes, f) != nbytes) die("truncated weights");
    fclose(f);

    Cursor c = { m->blob };
    int d = m->d, qz = m->quant;
    m->embed = take_mw(&c, m->vocab, d, qz);
    m->L = xcalloc(m->layers, sizeof(Layer));
    for (int l = 0; l < m->layers; l++) {
        Layer *L = &m->L[l];
        L->n1 = take(&c, d);
        L->q = take_mw(&c, d, d, qz);
        L->k = take_mw(&c, m->kvd, d, qz);
        L->v = take_mw(&c, m->kvd, d, qz);
        L->o = take_mw(&c, d, d, qz);
        L->gate = take_mw(&c, d, d, qz);
        L->qn = take(&c, m->hd);
        L->kn = take(&c, m->hd);
        L->n2 = take(&c, d);
        L->n3 = take(&c, d);
        L->n4 = take(&c, d);
        L->w1 = take_mw(&c, m->ffn, d, qz);
        L->w2 = take_mw(&c, d, m->ffn, qz);
        L->w3 = take_mw(&c, m->ffn, d, qz);
    }
    m->norm = take(&c, d);
    m->name_head = take_mw(&c, d, d, qz);
    if ((size_t)(c.p - m->blob) != nbytes) die("weights size mismatch");
    m->deq = xmalloc(sizeof(float) * (size_t)m->ffn * m->d);
    return m;
}

/* ------------------------------------------------------------- tokenizer */

#define PAD 0
#define BOS 1
#define EOS 2
#define UNK 3

static const char *SPECIALS[] = {
    "<pad>", "<bos>", "<eos>", "<unk>", "<tools>", "</tools>",
    "<query>", "</query>", "<call>", "</call>",
};
#define N_SPECIALS 10

typedef struct {
    uint32_t hash;
    char *key;
    int val;
} HEnt;

typedef struct {
    HEnt *ents;
    size_t cap;   /* power of two */
} HMap;

static uint32_t fnv1a(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h ? h : 1;
}

static void hmap_init(HMap *m, size_t want) {
    size_t cap = 16;
    while (cap < want * 2) cap <<= 1;
    m->cap = cap;
    m->ents = xcalloc(cap, sizeof(HEnt));
}

static void hmap_put(HMap *m, const char *key, int val) {
    uint32_t h = fnv1a(key, strlen(key));
    size_t i = h & (m->cap - 1);
    while (m->ents[i].hash) {
        if (m->ents[i].hash == h && strcmp(m->ents[i].key, key) == 0) {
            m->ents[i].val = val;
            return;
        }
        i = (i + 1) & (m->cap - 1);
    }
    m->ents[i].hash = h;
    m->ents[i].key = xstrdup(key);
    m->ents[i].val = val;
}

static int hmap_get(const HMap *m, const char *key, int dflt) {
    uint32_t h = fnv1a(key, strlen(key));
    size_t i = h & (m->cap - 1);
    while (m->ents[i].hash) {
        if (m->ents[i].hash == h && strcmp(m->ents[i].key, key) == 0)
            return m->ents[i].val;
        i = (i + 1) & (m->cap - 1);
    }
    return dflt;
}

typedef struct {
    int n_vocab;
    char **tok;       /* id -> utf-8 string */
    int *tok_clen;    /* id -> length in codepoints (Python len()) */
    HMap vocab;       /* string -> id */
    HMap ranks;       /* "a\x1fb" -> merge rank */
} Tok;

static int utf8_len(const char *s) {
    int n = 0;
    for (; *s; s++) if (((unsigned char)*s & 0xC0) != 0x80) n++;
    return n;
}

static Tok *tok_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open tokenizer file");
    uint32_t magic, ver, n;
    if (fread(&magic, 4, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        magic != 0x4B4F5454u || ver != 1)
        die("bad tokenizer header");
    if (fread(&n, 4, 1, f) != 1) die("bad tokenizer header");
    Tok *t = xcalloc(1, sizeof(Tok));
    t->n_vocab = (int)n;
    t->tok = xcalloc(n, sizeof(char *));
    t->tok_clen = xcalloc(n, sizeof(int));
    hmap_init(&t->vocab, n);
    char buf[65536];
    for (uint32_t i = 0; i < n; i++) {
        uint16_t len;
        if (fread(&len, 2, 1, f) != 1 || fread(buf, 1, len, f) != len)
            die("truncated tokenizer");
        buf[len] = 0;
        t->tok[i] = xstrdup(buf);
        t->tok_clen[i] = utf8_len(buf);
        hmap_put(&t->vocab, buf, (int)i);
    }
    uint32_t nm;
    if (fread(&nm, 4, 1, f) != 1) die("truncated tokenizer");
    hmap_init(&t->ranks, nm);
    char key[131072];
    for (uint32_t r = 0; r < nm; r++) {
        uint16_t la, lb;
        if (fread(&la, 2, 1, f) != 1 || fread(key, 1, la, f) != la) die("truncated merges");
        key[la] = 0x1f;
        if (fread(&lb, 2, 1, f) != 1 || fread(key + la + 1, 1, lb, f) != lb) die("truncated merges");
        key[la + 1 + lb] = 0;
        hmap_put(&t->ranks, key, (int)r);
    }
    fclose(f);
    return t;
}

/* Encode one pretokenized word by BPE merges. Mirrors _encode_word_uncached:
 * whole-word fast path, then greedy lowest-rank pair merge over CODEPOINTS. */
static int tok_encode_word(const Tok *t, const char *word, int *out, int max_out) {
    int whole = hmap_get(&t->vocab, word, -1);
    if (whole >= 0) {
        if (max_out < 1) die("token buffer overflow");
        out[0] = whole;
        return 1;
    }
    /* split into codepoint strings */
    int cap = utf8_len(word);
    if (cap == 0) return 0;
    char **parts = xmalloc(sizeof(char *) * cap);
    int np = 0;
    for (const char *p = word; *p;) {
        const char *q = p + 1;
        while (((unsigned char)*q & 0xC0) == 0x80) q++;
        parts[np++] = xstrndup(p, (size_t)(q - p));
        p = q;
    }
    char key[131072];
    while (np >= 2) {
        int best_rank = -1, best_i = -1;
        for (int i = 0; i + 1 < np; i++) {
            size_t la = strlen(parts[i]), lb = strlen(parts[i + 1]);
            memcpy(key, parts[i], la);
            key[la] = 0x1f;
            memcpy(key + la + 1, parts[i + 1], lb);
            key[la + 1 + lb] = 0;
            int r = hmap_get(&t->ranks, key, -1);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best_i = i; }
        }
        if (best_rank < 0) break;
        size_t la = strlen(parts[best_i]), lb = strlen(parts[best_i + 1]);
        char *merged = xmalloc(la + lb + 1);
        memcpy(merged, parts[best_i], la);
        memcpy(merged + la, parts[best_i + 1], lb + 1);
        free(parts[best_i]);
        free(parts[best_i + 1]);
        parts[best_i] = merged;
        memmove(parts + best_i + 1, parts + best_i + 2, sizeof(char *) * (np - best_i - 2));
        np--;
    }
    int n = 0;
    for (int i = 0; i < np; i++) {
        if (n >= max_out) die("token buffer overflow");
        out[n++] = hmap_get(&t->vocab, parts[i], UNK);
        free(parts[i]);
    }
    free(parts);
    return n;
}

static int is_structural(char c) {
    return c == '{' || c == '}' || c == '[' || c == ']' || c == ',' || c == ':' || c == '"';
}

static int is_hspace(char c) {
    /* [^\S\n]: whitespace that is not newline (ASCII) */
    return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
}

/* Pretokenize + encode one special-free segment (mirrors pretokenize()). */
static int tok_encode_segment(const Tok *t, const char *s, size_t n, int *out, int max_out) {
    int total = 0;
    char word[65536];
    size_t i = 0;
    while (i < n) {
        char c = s[i];
        size_t j;
        if (is_structural(c)) {
            j = i + 1;
        } else if (c == '\n') {
            j = i + 1;
        } else if (is_hspace(c)) {
            j = i + 1;
            while (j < n && is_hspace(s[j])) j++;
        } else {
            j = i + 1;
            while (j < n && !is_structural(s[j]) && s[j] != '\n' && !is_hspace(s[j])) j++;
        }
        if (j - i >= sizeof(word)) die("word too long");
        memcpy(word, s + i, j - i);
        word[j - i] = 0;
        total += tok_encode_word(t, word, out + total, max_out - total);
        i = j;
    }
    return total;
}

/* Full encode with special-token splitting (mirrors BPETokenizer.encode). */
static int tok_encode(const Tok *t, const char *s, int *out, int max_out) {
    int total = 0;
    size_t n = strlen(s), seg = 0, i = 0;
    while (i < n) {
        int sp = -1;
        size_t sp_len = 0;
        for (int k = 0; k < N_SPECIALS; k++) {
            size_t l = strlen(SPECIALS[k]);
            if (i + l <= n && memcmp(s + i, SPECIALS[k], l) == 0) { sp = k; sp_len = l; break; }
        }
        if (sp >= 0) {
            total += tok_encode_segment(t, s + seg, i - seg, out + total, max_out - total);
            int id = hmap_get(&t->vocab, SPECIALS[sp], -1);
            if (id < 0) die("special token missing from vocab");
            if (total >= max_out) die("token buffer overflow");
            out[total++] = id;
            i += sp_len;
            seg = i;
        } else {
            i++;
        }
    }
    total += tok_encode_segment(t, s + seg, n - seg, out + total, max_out - total);
    return total;
}

static const char *tok_str(const Tok *t, int id) {
    return (id >= 0 && id < t->n_vocab) ? t->tok[id] : "";
}

/* ----------------------------------------------------------- mini JSON */

typedef enum { J_NULL, J_BOOL, J_INT, J_FLOAT, J_STR, J_ARR, J_OBJ } JType;

typedef struct JVal {
    JType type;
    int b;
    long long i;
    double f;
    char *s;                 /* string value */
    struct JVal **items;     /* array items / object values */
    char **keys;             /* object keys */
    int n;
} JVal;

typedef struct { const char *s; size_t i, n; } JP;

static void jp_ws(JP *p) { while (p->i < p->n && isspace((unsigned char)p->s[p->i])) p->i++; }

static JVal *jparse_val(JP *p);

static char *jparse_string(JP *p) {
    if (p->s[p->i] != '"') die("json: expected string");
    p->i++;
    size_t cap = 32, len = 0;
    char *out = xmalloc(cap);
    while (p->i < p->n && p->s[p->i] != '"') {
        char c = p->s[p->i++];
        if (c == '\\') {
            char e = p->s[p->i++];
            switch (e) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case '/': c = '/'; break;
                case '\\': c = '\\'; break;
                case '"': c = '"'; break;
                case 'u': {
                    unsigned cp = 0;
                    for (int k = 0; k < 4; k++) {
                        char h = p->s[p->i++];
                        cp = cp * 16 + (unsigned)(h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
                    }
                    if (cp >= 0xD800 && cp < 0xDC00 && p->s[p->i] == '\\' && p->s[p->i + 1] == 'u') {
                        p->i += 2;
                        unsigned lo = 0;
                        for (int k = 0; k < 4; k++) {
                            char h = p->s[p->i++];
                            lo = lo * 16 + (unsigned)(h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    char tmp[4];
                    int tn = 0;
                    if (cp < 0x80) tmp[tn++] = (char)cp;
                    else if (cp < 0x800) {
                        tmp[tn++] = (char)(0xC0 | (cp >> 6));
                        tmp[tn++] = (char)(0x80 | (cp & 63));
                    } else if (cp < 0x10000) {
                        tmp[tn++] = (char)(0xE0 | (cp >> 12));
                        tmp[tn++] = (char)(0x80 | ((cp >> 6) & 63));
                        tmp[tn++] = (char)(0x80 | (cp & 63));
                    } else {
                        tmp[tn++] = (char)(0xF0 | (cp >> 18));
                        tmp[tn++] = (char)(0x80 | ((cp >> 12) & 63));
                        tmp[tn++] = (char)(0x80 | ((cp >> 6) & 63));
                        tmp[tn++] = (char)(0x80 | (cp & 63));
                    }
                    for (int k = 0; k < tn; k++) {
                        if (len + 1 >= cap) { cap *= 2; out = realloc(out, cap); }
                        out[len++] = tmp[k];
                    }
                    continue;
                }
                default: c = e;
            }
        }
        if (len + 1 >= cap) { cap *= 2; out = realloc(out, cap); }
        out[len++] = c;
    }
    p->i++;
    out[len] = 0;
    return out;
}

static JVal *jnew(JType t) {
    JVal *v = xcalloc(1, sizeof(JVal));
    v->type = t;
    return v;
}

static JVal *jparse_val(JP *p) {
    jp_ws(p);
    if (p->i >= p->n) die("json: eof");
    char c = p->s[p->i];
    if (c == '{') {
        p->i++;
        JVal *v = jnew(J_OBJ);
        jp_ws(p);
        if (p->s[p->i] == '}') { p->i++; return v; }
        for (;;) {
            jp_ws(p);
            char *key = jparse_string(p);
            jp_ws(p);
            if (p->s[p->i++] != ':') die("json: expected :");
            JVal *val = jparse_val(p);
            v->keys = realloc(v->keys, sizeof(char *) * (v->n + 1));
            v->items = realloc(v->items, sizeof(JVal *) * (v->n + 1));
            v->keys[v->n] = key;
            v->items[v->n] = val;
            v->n++;
            jp_ws(p);
            if (p->s[p->i] == ',') { p->i++; continue; }
            if (p->s[p->i] == '}') { p->i++; return v; }
            die("json: expected , or }");
        }
    }
    if (c == '[') {
        p->i++;
        JVal *v = jnew(J_ARR);
        jp_ws(p);
        if (p->s[p->i] == ']') { p->i++; return v; }
        for (;;) {
            JVal *val = jparse_val(p);
            v->items = realloc(v->items, sizeof(JVal *) * (v->n + 1));
            v->items[v->n++] = val;
            jp_ws(p);
            if (p->s[p->i] == ',') { p->i++; continue; }
            if (p->s[p->i] == ']') { p->i++; return v; }
            die("json: expected , or ]");
        }
    }
    if (c == '"') {
        JVal *v = jnew(J_STR);
        v->s = jparse_string(p);
        return v;
    }
    if (strncmp(p->s + p->i, "true", 4) == 0) { p->i += 4; JVal *v = jnew(J_BOOL); v->b = 1; return v; }
    if (strncmp(p->s + p->i, "false", 5) == 0) { p->i += 5; JVal *v = jnew(J_BOOL); v->b = 0; return v; }
    if (strncmp(p->s + p->i, "null", 4) == 0) { p->i += 4; return jnew(J_NULL); }
    /* number */
    size_t start = p->i;
    while (p->i < p->n && (isdigit((unsigned char)p->s[p->i]) || strchr("+-.eE", p->s[p->i])))
        p->i++;
    char *num = xstrndup(p->s + start, p->i - start);
    JVal *v;
    if (strpbrk(num, ".eE")) {
        v = jnew(J_FLOAT);
        v->f = strtod(num, NULL);
    } else {
        v = jnew(J_INT);
        v->i = strtoll(num, NULL, 10);
    }
    free(num);
    return v;
}

static JVal *jparse(const char *s) {
    JP p = { s, 0, strlen(s) };
    return jparse_val(&p);
}

static JVal *jget(const JVal *obj, const char *key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    for (int i = 0; i < obj->n; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

static const char *jstr(const JVal *v, const char *dflt) {
    return (v && v->type == J_STR) ? v->s : dflt;
}

/* ------------------------------------------------- number formatting */

/* Python repr() of a float: shortest string that round-trips. */
static void fmt_double(double x, char *out) {
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(out, 64, "%.*g", prec, x);
        if (strtod(out, NULL) == x) break;
    }
    /* Python floats always show a decimal point or exponent */
    if (!strpbrk(out, ".eEnN")) strcat(out, ".0");
}

/* Python str() of a JSON scalar (for enum options). */
static char *jval_str(const JVal *v) {
    char buf[64];
    switch (v->type) {
        case J_STR: return xstrdup(v->s);
        case J_INT: snprintf(buf, 64, "%lld", v->i); return xstrdup(buf);
        case J_FLOAT: fmt_double(v->f, buf); return xstrdup(buf);
        case J_BOOL: return xstrdup(v->b ? "True" : "False");
        default: return xstrdup("null");
    }
}

/* ------------------------------------------------------ catalog / tools */

typedef struct {
    char *key;
    char *type;       /* string/integer/number/boolean/... */
    char *desc;
    JVal *enum_vals;  /* NULL if none */
    int required;
} Param;

typedef struct {
    char *name;
    char *desc;
    Param *params;    /* sorted by key (Python sorted(props)) */
    int n_params;
} Tool;

static int cmp_param(const void *a, const void *b) {
    return strcmp(((const Param *)a)->key, ((const Param *)b)->key);
}

static Tool *tools_from_json(const JVal *arr, int *n_out) {
    if (!arr || arr->type != J_ARR) die("catalog: expected array of tools");
    Tool *tools = xcalloc(arr->n, sizeof(Tool));
    for (int i = 0; i < arr->n; i++) {
        const JVal *t = arr->items[i];
        Tool *T = &tools[i];
        T->name = xstrdup(jstr(jget(t, "name"), ""));
        T->desc = xstrdup(jstr(jget(t, "description"), ""));
        const JVal *params = jget(t, "parameters");
        const JVal *props = params ? jget(params, "properties") : NULL;
        const JVal *req = params ? jget(params, "required") : NULL;
        int np = (props && props->type == J_OBJ) ? props->n : 0;
        T->params = xcalloc(np ? np : 1, sizeof(Param));
        T->n_params = np;
        for (int k = 0; k < np; k++) {
            Param *P = &T->params[k];
            P->key = xstrdup(props->keys[k]);
            const JVal *spec = props->items[k];
            P->type = xstrdup(jstr(jget(spec, "type"), "string"));
            P->desc = xstrdup(jstr(jget(spec, "description"), ""));
            const JVal *en = jget(spec, "enum");
            P->enum_vals = (en && en->type == J_ARR && en->n > 0) ? (JVal *)en : NULL;
            P->required = 0;
            if (req && req->type == J_ARR)
                for (int r = 0; r < req->n; r++)
                    if (req->items[r]->type == J_STR && strcmp(req->items[r]->s, P->key) == 0)
                        P->required = 1;
        }
        qsort(T->params, T->n_params, sizeof(Param), cmp_param);
    }
    *n_out = arr->n;
    return tools;
}

/* --------------------------------------------------------- prompt render */

typedef struct { char *buf; size_t len, cap; } SB;

static void sb_init(SB *b) { b->cap = 256; b->len = 0; b->buf = xmalloc(b->cap); b->buf[0] = 0; }

static void sb_put(SB *b, const char *s) {
    size_t n = strlen(s);
    while (b->len + n + 1 > b->cap) { b->cap *= 2; b->buf = realloc(b->buf, b->cap); }
    memcpy(b->buf + b->len, s, n + 1);
    b->len += n;
}

static const char *TYPE_ABBR(const char *t) {
    if (strcmp(t, "string") == 0) return "str";
    if (strcmp(t, "integer") == 0) return "int";
    if (strcmp(t, "number") == 0) return "num";
    if (strcmp(t, "boolean") == 0) return "bool";
    if (strcmp(t, "object") == 0) return "obj";
    if (strcmp(t, "array") == 0) return "arr";
    return t;
}

/* mirrors render.tool_signature + prompt_text */
static char *render_prompt(const char *query, const Tool *tools, int n_tools) {
    SB b;
    sb_init(&b);
    sb_put(&b, "<tools>\n");
    for (int i = 0; i < n_tools; i++) {
        const Tool *t = &tools[i];
        sb_put(&b, "- ");
        sb_put(&b, t->name);
        sb_put(&b, " (");
        for (int k = 0; k < t->n_params; k++) {
            const Param *p = &t->params[k];
            if (k) sb_put(&b, " ");
            sb_put(&b, p->key);
            sb_put(&b, "=");
            if (p->enum_vals) {
                sb_put(&b, "enum(");
                for (int e = 0; e < p->enum_vals->n; e++) {
                    if (e) sb_put(&b, "|");
                    char *es = jval_str(p->enum_vals->items[e]);
                    sb_put(&b, es);
                    free(es);
                }
                sb_put(&b, ")");
            } else {
                sb_put(&b, TYPE_ABBR(p->type));
            }
            if (p->required) sb_put(&b, "!");
        }
        sb_put(&b, ") ");
        /* python: t.get("description","").strip() */
        char *d = xstrdup(t->desc);
        char *s0 = d;
        while (*s0 && isspace((unsigned char)*s0)) s0++;
        char *e0 = s0 + strlen(s0);
        while (e0 > s0 && isspace((unsigned char)e0[-1])) e0--;
        *e0 = 0;
        sb_put(&b, s0);
        free(d);
        sb_put(&b, "\n");
    }
    sb_put(&b, "</tools>\n<query>\n");
    sb_put(&b, query);
    sb_put(&b, "\n</query>\n<call>\n");
    return b.buf;
}

/* -------------------------------------------------- lexical retrieval */

/* _tok(): camel-split, lowercase, [a-z0-9]+ runs, dedupe. */
typedef struct { char **w; int n; } WordSet;

static void wordset_add(WordSet *ws, const char *w) {
    for (int i = 0; i < ws->n; i++)
        if (strcmp(ws->w[i], w) == 0) return;
    ws->w = realloc(ws->w, sizeof(char *) * (ws->n + 1));
    ws->w[ws->n++] = xstrdup(w);
}

static void wordset_free(WordSet *ws) {
    for (int i = 0; i < ws->n; i++) free(ws->w[i]);
    free(ws->w);
    ws->w = NULL;
    ws->n = 0;
}

static int wordset_has(const WordSet *ws, const char *w) {
    for (int i = 0; i < ws->n; i++)
        if (strcmp(ws->w[i], w) == 0) return 1;
    return 0;
}

static void tok_words(const char *text, WordSet *ws) {
    /* insert breaks at (?<=[a-z0-9])(?=[A-Z]) and (?<=[A-Z])(?=[A-Z][a-z]), then
     * lowercase and take [a-z0-9]+ runs */
    size_t n = strlen(text);
    char *tmp = xmalloc(2 * n + 1);
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        char prev = i ? text[i - 1] : 0;
        char cur = text[i];
        char nxt = i + 1 < n ? text[i + 1] : 0;
        if (i && ((islower((unsigned char)prev) || isdigit((unsigned char)prev)) && isupper((unsigned char)cur)))
            tmp[o++] = ' ';
        else if (i && isupper((unsigned char)prev) && isupper((unsigned char)cur) && islower((unsigned char)nxt))
            tmp[o++] = ' ';
        tmp[o++] = cur;
    }
    tmp[o] = 0;
    char word[4096];
    int wl = 0;
    for (size_t i = 0; i <= o; i++) {
        char c = i < o ? (char)tolower((unsigned char)tmp[i]) : 0;
        if (c && ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) {
            if (wl < 4095) word[wl++] = c;
        } else if (wl) {
            word[wl] = 0;
            wordset_add(ws, word);
            wl = 0;
        }
    }
    free(tmp);
}

static int wordset_overlap(const WordSet *a, const WordSet *b) {
    int n = 0;
    for (int i = 0; i < a->n; i++)
        if (wordset_has(b, a->w[i])) n++;
    return n;
}

typedef struct {
    char *name;
    JVal *args;   /* J_OBJ of emitted arguments (values as JVal) */
} Call;

static void bag_from_query(const char *query, const Call *emitted, int n_emitted, WordSet *bag) {
    tok_words(query, bag);
    for (int c = 0; c < n_emitted; c++) {
        const JVal *args = emitted[c].args;
        for (int i = 0; i < args->n; i++) {
            char *vs = jval_str(args->items[i]);   /* str(v) */
            tok_words(vs, bag);
            free(vs);
        }
    }
}

static void tool_text_words(const Tool *t, WordSet *ws) {
    /* tool_text: name + description + property keys, joined by spaces */
    tok_words(t->name, ws);
    tok_words(t->desc, ws);
    for (int k = 0; k < t->n_params; k++) tok_words(t->params[k].key, ws);
}

/* retrieve(): returns indices into tools, at most k */
static int retrieve_tools(const char *query, const Tool *tools, int n_tools,
                          int k, const Call *emitted, int n_emitted, int *out_idx) {
    WordSet bag = {0};
    bag_from_query(query, emitted, n_emitted, &bag);
    double *score = xmalloc(sizeof(double) * n_tools);
    for (int i = 0; i < n_tools; i++) {
        WordSet tw = {0};
        tool_text_words(&tools[i], &tw);
        double s = wordset_overlap(&bag, &tw);
        if (wordset_has(&bag, tools[i].name)) s += 0.25;   /* name in bag */
        for (int c = 0; c < n_emitted; c++)
            if (strcmp(emitted[c].name, tools[i].name) == 0) { s -= 1.5; break; }
        score[i] = s;
        wordset_free(&tw);
    }
    /* sort indices by (score desc, index asc) */
    int *idx = xmalloc(sizeof(int) * n_tools);
    for (int i = 0; i < n_tools; i++) idx[i] = i;
    for (int i = 1; i < n_tools; i++) {
        int v = idx[i];
        int j = i - 1;
        while (j >= 0 && (score[idx[j]] < score[v] ||
               (score[idx[j]] == score[v] && idx[j] > v))) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = v;
    }
    int n = k < n_tools ? k : n_tools;
    for (int i = 0; i < n; i++) out_idx[i] = idx[i];
    free(idx);
    free(score);
    wordset_free(&bag);
    return n;
}

/* lexical_scores(): normalized in [0,1] over the given tools */
static void lexical_scores_c(const char *query, const Tool *tools, const int *tool_idx,
                             int n, const Call *emitted, int n_emitted, double *out) {
    WordSet bag = {0};
    bag_from_query(query, emitted, n_emitted, &bag);
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        const Tool *t = &tools[tool_idx[i]];
        WordSet tw = {0}, nw = {0};
        tool_text_words(t, &tw);
        tok_words(t->name, &nw);
        double s = wordset_overlap(&bag, &tw) + 2.0 * wordset_overlap(&bag, &nw);
        for (int c = 0; c < n_emitted; c++)
            if (strcmp(emitted[c].name, t->name) == 0) { s -= 1.0; break; }
        if (s < 0) s = 0;
        out[i] = s;
        total += s;
        wordset_free(&tw);
        wordset_free(&nw);
    }
    wordset_free(&bag);
    if (total <= 0) {
        double u = 1.0 / (n > 1 ? n : 1);
        for (int i = 0; i < n; i++) out[i] = u;
    } else {
        for (int i = 0; i < n; i++) out[i] /= total;
    }
}

/* --------------------------------------------------------- decoder core */

#define MAX_CALLS 4
#define LEX_MAX_WEIGHT 0.85
#define LEX_SHARPNESS 4.0
#define REFUSE_GATE 0.35
#define MAX_VALUE_TOKENS 96
#define MAX_SEQ_BUF 2048

typedef struct {
    Model *m;
    Tok *t;
    float *kc, *vc;        /* [layers][MAX_SEQ_BUF][kvd] */
    float *hidden;         /* [MAX_SEQ_BUF][d] post-final-norm */
    int *ids;
    int n_ids;             /* == cache length */
    float *logits;         /* [vocab], for the last fed token */
    int logits_valid;
    /* chunk scratch, all [MAX_SEQ_BUF][*] */
    float *x, *xn, *q, *k, *v, *y, *g, *h1, *h3, *ff, *att;
    /* rope tables [MAX_SEQ_BUF][hd] */
    float *rcos, *rsin;
} Dec;

static void dec_free(Dec *d) {
    free(d->kc); free(d->vc); free(d->hidden); free(d->ids); free(d->logits);
    free(d->x); free(d->xn); free(d->q); free(d->k); free(d->v); free(d->y);
    free(d->g); free(d->h1); free(d->h3); free(d->ff); free(d->att);
    free(d->rcos); free(d->rsin);
    free(d);
}

static Dec *dec_new(Model *m, Tok *t) {
    Dec *d = xcalloc(1, sizeof(Dec));
    d->m = m;
    d->t = t;
    size_t S = MAX_SEQ_BUF;
    d->kc = xmalloc(sizeof(float) * m->layers * S * m->kvd);
    d->vc = xmalloc(sizeof(float) * m->layers * S * m->kvd);
    d->hidden = xmalloc(sizeof(float) * S * m->d);
    d->ids = xmalloc(sizeof(int) * S);
    d->logits = xmalloc(sizeof(float) * m->vocab);
    d->x = xmalloc(sizeof(float) * S * m->d);
    d->xn = xmalloc(sizeof(float) * S * m->d);
    d->q = xmalloc(sizeof(float) * S * m->d);
    d->k = xmalloc(sizeof(float) * S * m->kvd);
    d->v = xmalloc(sizeof(float) * S * m->kvd);
    d->y = xmalloc(sizeof(float) * S * m->d);
    d->g = xmalloc(sizeof(float) * S * m->d);
    d->h1 = xmalloc(sizeof(float) * S * m->ffn);
    d->h3 = xmalloc(sizeof(float) * S * m->ffn);
    d->ff = xmalloc(sizeof(float) * S * m->d);
    d->att = xmalloc(sizeof(float) * S);
    d->rcos = xmalloc(sizeof(float) * S * m->hd);
    d->rsin = xmalloc(sizeof(float) * S * m->hd);
    int half = m->hd / 2;
    for (size_t pos = 0; pos < S; pos++) {
        for (int j = 0; j < half; j++) {
            double freq = pow((double)m->theta, -2.0 * j / m->hd);
            double a = (double)pos * freq;
            d->rcos[pos * m->hd + j] = d->rcos[pos * m->hd + j + half] = (float)cos(a);
            d->rsin[pos * m->hd + j] = d->rsin[pos * m->hd + j + half] = (float)sin(a);
        }
    }
    return d;
}

/* Forward a chunk of n tokens in one pass (sgemm across rows — same math as
 * Python's multi-token feed, and where all the prefill speed comes from).
 * Updates KV cache and hidden[]; logits are NOT projected here. */
static void dec_forward_chunk(Dec *d, const int *ids, int n) {
    Model *m = d->m;
    int D = m->d, HD = m->hd, H = m->heads, KV = m->kv, rep = H / KV, FFN = m->ffn;
    int pos0 = d->n_ids;
    if (pos0 + n > MAX_SEQ_BUF) die("sequence too long");
    int half = HD / 2;
    for (int r = 0; r < n; r++)
        mw_row(&m->embed, ids[r], d->x + (size_t)r * D);
    for (int l = 0; l < m->layers; l++) {
        Layer *L = &m->L[l];
        for (int r = 0; r < n; r++)
            rmsnorm(d->x + (size_t)r * D, L->n1, d->xn + (size_t)r * D, D);
        mw_matmul(m, &L->q, d->xn, d->q, n);
        mw_matmul(m, &L->k, d->xn, d->k, n);
        mw_matmul(m, &L->v, d->xn, d->v, n);
        for (int r = 0; r < n; r++) {
            int pos = pos0 + r;
            const float *cosv = d->rcos + (size_t)pos * HD;
            const float *sinv = d->rsin + (size_t)pos * HD;
            for (int h = 0; h < H; h++) {
                float *qh = d->q + (size_t)r * D + h * HD;
                float tmp[128];
                rmsnorm(qh, L->qn, tmp, HD);
                for (int j = 0; j < half; j++) {
                    float a = tmp[j], b = tmp[j + half];
                    qh[j] = a * cosv[j] - b * sinv[j];
                    qh[j + half] = b * cosv[j] + a * sinv[j];
                }
            }
            float *krow = d->k + (size_t)r * m->kvd;
            for (int h = 0; h < KV; h++) {
                float *kh = krow + h * HD;
                float tmp[128];
                rmsnorm(kh, L->kn, tmp, HD);
                for (int j = 0; j < half; j++) {
                    float a = tmp[j], b = tmp[j + half];
                    kh[j] = a * cosv[j] - b * sinv[j];
                    kh[j + half] = b * cosv[j] + a * sinv[j];
                }
            }
            memcpy(d->kc + ((size_t)l * MAX_SEQ_BUF + pos) * m->kvd, krow, sizeof(float) * m->kvd);
            memcpy(d->vc + ((size_t)l * MAX_SEQ_BUF + pos) * m->kvd,
                   d->v + (size_t)r * m->kvd, sizeof(float) * m->kvd);
        }
        float scale = 1.0f / sqrtf((float)HD);
        for (int r = 0; r < n; r++) {
            int ctx = pos0 + r + 1;   /* causal: row r sees cache[0..pos0+r] */
            for (int h = 0; h < H; h++) {
                int kvh = h / rep;
                const float *qh = d->q + (size_t)r * D + h * HD;
                for (int tpos = 0; tpos < ctx; tpos++) {
                    const float *kt = d->kc + ((size_t)l * MAX_SEQ_BUF + tpos) * m->kvd + kvh * HD;
                    float acc = 0.0f;
                    for (int j = 0; j < HD; j++) acc += qh[j] * kt[j];
                    d->att[tpos] = acc * scale;
                }
                softmax_(d->att, ctx);
                float *yh = d->y + (size_t)r * D + h * HD;
                memset(yh, 0, sizeof(float) * HD);
                for (int tpos = 0; tpos < ctx; tpos++) {
                    const float *vt = d->vc + ((size_t)l * MAX_SEQ_BUF + tpos) * m->kvd + kvh * HD;
                    float p = d->att[tpos];
                    for (int j = 0; j < HD; j++) yh[j] += p * vt[j];
                }
            }
        }
        mw_matmul(m, &L->gate, d->xn, d->g, n);
        for (size_t j = 0; j < (size_t)n * D; j++)
            d->y[j] *= 1.0f / (1.0f + expf(-d->g[j]));
        mw_matmul(m, &L->o, d->y, d->g, n);            /* g = attn out */
        for (int r = 0; r < n; r++) {
            rmsnorm(d->g + (size_t)r * D, L->n2, d->xn + (size_t)r * D, D);
            float *xr = d->x + (size_t)r * D;
            const float *nr = d->xn + (size_t)r * D;
            for (int j = 0; j < D; j++) xr[j] += nr[j];
            rmsnorm(xr, L->n3, d->xn + (size_t)r * D, D);
        }
        mw_matmul(m, &L->w1, d->xn, d->h1, n);
        mw_matmul(m, &L->w3, d->xn, d->h3, n);
        for (size_t j = 0; j < (size_t)n * FFN; j++) {
            float s = d->h1[j] / (1.0f + expf(-d->h1[j]));   /* silu */
            d->h1[j] = s * d->h3[j];
        }
        mw_matmul(m, &L->w2, d->h1, d->ff, n);
        for (int r = 0; r < n; r++) {
            rmsnorm(d->ff + (size_t)r * D, L->n4, d->xn + (size_t)r * D, D);
            float *xr = d->x + (size_t)r * D;
            const float *nr = d->xn + (size_t)r * D;
            for (int j = 0; j < D; j++) xr[j] += nr[j];
        }
    }
    for (int r = 0; r < n; r++)
        rmsnorm(d->x + (size_t)r * D, m->norm,
                d->hidden + (size_t)(pos0 + r) * m->d, D);
    for (int r = 0; r < n; r++) d->ids[d->n_ids++] = ids[r];
    g_forwards += n;
}

/* Logits are projected LAZILY on first read after a feed. Python projects
 * eagerly on every feed, but from the same hidden state, so reads see
 * identical values; feeds whose logits are never read (most structural feeds)
 * skip the |vocab| x d sweep entirely. */
static void dec_feed_ids(Dec *d, const int *ids, int n) {
    if (!n) return;
    dec_forward_chunk(d, ids, n);
    d->logits_valid = 0;
}

static void ensure_logits(Dec *d) {
    if (d->logits_valid) return;
    mw_matvec(&d->m->embed, d->hidden + (size_t)(d->n_ids - 1) * d->m->d, d->logits);
    d->logits_valid = 1;
}

static void dec_feed_str(Dec *d, const char *s) {
    int ids[8192];
    int n = tok_encode(d->t, s, ids, 8192);
    dec_feed_ids(d, ids, n);
}

static void dec_feed_id(Dec *d, int id) { dec_feed_ids(d, &id, 1); }

typedef struct { int n_ids; float *logits; int logits_valid; } Snap;

static Snap dec_snapshot(Dec *d) {
    Snap s;
    ensure_logits(d);   /* rollback must restore readable logits */
    s.n_ids = d->n_ids;
    s.logits_valid = d->logits_valid;
    s.logits = xmalloc(sizeof(float) * d->m->vocab);
    memcpy(s.logits, d->logits, sizeof(float) * d->m->vocab);
    return s;
}

static void dec_rollback(Dec *d, const Snap *s) {
    d->n_ids = s->n_ids;   /* truncates ids, caches, hiddens in one move */
    memcpy(d->logits, s->logits, sizeof(float) * d->m->vocab);
    d->logits_valid = s->logits_valid;
}

static void snap_free(Snap *s) { free(s->logits); }

static int first_token_of(Dec *d, const char *s) {
    int ids[64];
    int n = tok_encode(d->t, s, ids, 64);
    if (!n) die("empty option");
    return ids[0];
}

/* score_first: per-option first-token logprob */
static void dec_score_first(Dec *d, const char **opts, int n, double *out) {
    ensure_logits(d);
    double lse = logsumexp(d->logits, d->m->vocab);
    for (int i = 0; i < n; i++)
        out[i] = (double)d->logits[first_token_of(d, opts[i])] - lse;
}

/* choose_first: argmax over first-token logprobs */
static int dec_choose_first(Dec *d, const char **opts, int n) {
    double lp[64];
    dec_score_first(d, opts, n, lp);
    int best = 0;
    for (int i = 1; i < n; i++) if (lp[i] > lp[best]) best = i;
    return best;
}

/* score_str: length-normalized teacher-forced logprob, rollback after each */
static void dec_score_str(Dec *d, const char **opts, int n, double *out) {
    Snap snap = dec_snapshot(d);
    for (int i = 0; i < n; i++) {
        int ids[8192];
        int ni = tok_encode(d->t, opts[i], ids, 8192);
        double lp = 0.0;
        for (int j = 0; j < ni; j++) {
            ensure_logits(d);
            lp += (double)d->logits[ids[j]] - logsumexp(d->logits, d->m->vocab);
            dec_feed_id(d, ids[j]);
        }
        dec_rollback(d, &snap);
        out[i] = lp / (ni > 1 ? ni : 1);
    }
    snap_free(&snap);
}

static int dec_choose_str(Dec *d, const char **opts, int n) {
    double sc[512];
    if (n > 512) die("too many options");
    dec_score_str(d, opts, n, sc);
    int best = 0;
    for (int i = 1; i < n; i++) if (sc[i] > sc[best]) best = i;
    return best;
}

/* gen_string_value: free-generate until closing quote; ban specials; break
 * period-k loops (k in 2,3,4 with three consecutive repeats) */
static char *dec_gen_string(Dec *d) {
    int close_id = hmap_get(&d->t->vocab, "\"", -1);
    if (close_id < 0) die("no quote token");
    int banned[16];
    int n_banned = 0;
    for (int i = 0; i < 4; i++) banned[n_banned++] = i;   /* pad/bos/eos/unk */
    for (int k = 4; k < N_SPECIALS; k++) {
        int id = hmap_get(&d->t->vocab, SPECIALS[k], -1);
        if (id >= 0) banned[n_banned++] = id;
    }
    SB out;
    sb_init(&out);
    int ids_out[MAX_VALUE_TOKENS + 1];
    int n_out = 0;
    for (int step = 0; step < MAX_VALUE_TOKENS; step++) {
        ensure_logits(d);
        /* argmax over masked logits without copying the whole vector */
        int nxt = -1;
        float best = -1e30f;
        for (int i = 0; i < d->m->vocab; i++) {
            int skip = 0;
            for (int b = 0; b < n_banned; b++) if (i == banned[b]) { skip = 1; break; }
            if (skip) continue;
            if (d->logits[i] > best) { best = d->logits[i]; nxt = i; }
        }
        if (nxt == close_id) break;
        int looped = 0;
        ids_out[n_out] = nxt;
        int cn = n_out + 1;
        for (int k = 2; k <= 4; k++) {
            if (cn >= 3 * k) {
                const int *c = ids_out;
                if (memcmp(c + cn - k, c + cn - 2 * k, k * sizeof(int)) == 0 &&
                    memcmp(c + cn - 2 * k, c + cn - 3 * k, k * sizeof(int)) == 0) {
                    looped = 1;
                    break;
                }
            }
        }
        if (looped) break;
        dec_feed_id(d, nxt);
        n_out++;
        sb_put(&out, tok_str(d->t, nxt));
    }
    return out.buf;
}

/* is this string a valid Python float() literal? (for gen_number candidates) */
static int is_pyfloat(const char *s) {
    if (!*s) return 0;
    char *end;
    strtod(s, &end);
    if (*end) return 0;
    /* strtod accepts "inf"/"nan" but candidates are [0-9.-] only; strtod also
     * accepts "." as 0 on some libcs — Python float(".") fails */
    if (strcmp(s, ".") == 0 || strcmp(s, "-") == 0 || strcmp(s, "-.") == 0) return 0;
    /* must contain a digit */
    for (const char *p = s; *p; p++) if (isdigit((unsigned char)*p)) return 1;
    return 0;
}

/* is this a valid JSON number (json.loads)? */
static int is_json_number(const char *s, double *out, int *is_int) {
    const char *p = s;
    if (*p == '-') p++;
    if (!isdigit((unsigned char)*p)) return 0;
    if (*p == '0' && isdigit((unsigned char)p[1])) return 0;   /* no leading zeros */
    while (isdigit((unsigned char)*p)) p++;
    int isint = 1;
    if (*p == '.') {
        isint = 0;
        p++;
        if (!isdigit((unsigned char)*p)) return 0;
        while (isdigit((unsigned char)*p)) p++;
    }
    if (*p == 'e' || *p == 'E') {
        isint = 0;
        p++;
        if (*p == '+' || *p == '-') p++;
        if (!isdigit((unsigned char)*p)) return 0;
        while (isdigit((unsigned char)*p)) p++;
    }
    if (*p) return 0;
    *out = strtod(s, NULL);
    *is_int = isint;
    return 1;
}

/* gen_number_value: mirrors the top-96-candidate scan and the structural-stop */
static char *dec_gen_number(Dec *d) {
    SB out;
    sb_init(&out);
    int comma_id = hmap_get(&d->t->vocab, ",", -1);
    int brace_id = hmap_get(&d->t->vocab, "}", -1);
    int *order = xmalloc(sizeof(int) * d->m->vocab);
    for (int step = 0; step < 6; step++) {
        ensure_logits(d);
        for (int i = 0; i < d->m->vocab; i++) order[i] = i;
        /* full argsort desc to mirror torch.argsort, then scan top 96 */
        const float *lg = d->logits;
        /* insertion of top-96 via partial selection sort: enough and exact */
        for (int i = 0; i < 96 && i < d->m->vocab; i++) {
            int best = i;
            for (int j = i + 1; j < d->m->vocab; j++)
                if (lg[order[j]] > lg[order[best]]) best = j;
            int tmp = order[i];
            order[i] = order[best];
            order[best] = tmp;
        }
        int nxt = -1;
        for (int c = 0; c < 96 && c < d->m->vocab; c++) {
            const char *s = tok_str(d->t, order[c]);
            if (!*s) continue;
            int ok = 1;
            for (const char *p = s; *p; p++)
                if (!strchr("0123456789.-", *p)) { ok = 0; break; }
            if (!ok) continue;
            char cand[4096];
            snprintf(cand, sizeof cand, "%s%s", out.buf, s);
            if (!is_pyfloat(cand)) continue;
            nxt = order[c];
            break;
        }
        if (nxt < 0) break;
        float best_struct = -1e30f;
        if (comma_id >= 0 && d->logits[comma_id] > best_struct) best_struct = d->logits[comma_id];
        if (brace_id >= 0 && d->logits[brace_id] > best_struct) best_struct = d->logits[brace_id];
        if (out.len > 0 && best_struct > d->logits[nxt]) break;
        sb_put(&out, tok_str(d->t, nxt));
        dec_feed_id(d, nxt);
    }
    free(order);
    if (out.len == 0) {
        dec_feed_str(d, "0");
        sb_put(&out, "0");
    }
    return out.buf;
}

/* ------------------------------------------------- name spans in prompt */

typedef struct { int start, end; int found; } SpanTok;

/* mirrors data.name_spans_in_prompt + demo's +1 BOS shift */
static void name_spans(const Tok *t, const char *prompt, const int *pids, int n_pids,
                       const Tool *tools, int n_tools, SpanTok *spans) {
    /* char offsets in CODEPOINTS, matching Python len() semantics */
    int *offs = xmalloc(sizeof(int) * (n_pids + 1));
    offs[0] = 0;
    for (int i = 0; i < n_pids; i++)
        offs[i + 1] = offs[i] + ((pids[i] >= 0 && pids[i] < t->n_vocab) ? t->tok_clen[pids[i]] : 0);
    for (int ti = 0; ti < n_tools; ti++) {
        spans[ti].found = 0;
        char marker[4096];
        snprintf(marker, sizeof marker, "- %s (", tools[ti].name);
        const char *hit = strstr(prompt, marker);
        if (!hit) continue;
        /* byte position -> codepoint position */
        int cpos = 0;
        for (const char *p = prompt; p < hit; p++)
            if (((unsigned char)*p & 0xC0) != 0x80) cpos++;
        int vs = cpos + 2;
        int ve = vs + utf8_len(tools[ti].name);
        /* char_to_tok binary search: first token whose span contains char c */
        int lo, hi;
        lo = 0; hi = n_pids - 1;
        int c = vs;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (offs[mid + 1] <= c) lo = mid + 1; else hi = mid;
        }
        int s_tok = lo;
        lo = 0; hi = n_pids - 1;
        c = ve - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (offs[mid + 1] <= c) lo = mid + 1; else hi = mid;
        }
        int e_tok = lo + 1;
        spans[ti].start = s_tok + 1;   /* +1: BOS shift (demo.py) */
        spans[ti].end = e_tok + 1;
        spans[ti].found = 1;
    }
    free(offs);
}

/* -------------------------------------------------- output (dumps_calls) */

static void json_escape_to(SB *b, const char *s) {
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"') sb_put(b, "\\\"");
        else if (*p == '\\') sb_put(b, "\\\\");
        else if (*p == '\n') sb_put(b, "\\n");
        else if (*p == '\t') sb_put(b, "\\t");
        else if (*p == '\r') sb_put(b, "\\r");
        else if (*p < 0x20) {
            char tmp[8];
            snprintf(tmp, 8, "\\u%04x", *p);
            sb_put(b, tmp);
        } else {
            char tmp[2] = { (char)*p, 0 };
            sb_put(b, tmp);
        }
    }
}

static void jval_dump(SB *b, const JVal *v) {
    char tmp[64];
    switch (v->type) {
        case J_STR:
            sb_put(b, "\"");
            json_escape_to(b, v->s);
            sb_put(b, "\"");
            break;
        case J_BOOL:
            sb_put(b, v->b ? "true" : "false");
            break;
        case J_INT:
            snprintf(tmp, 64, "%lld", v->i);
            sb_put(b, tmp);
            break;
        case J_FLOAT:
            fmt_double(v->f, tmp);
            sb_put(b, tmp);
            break;
        default:
            sb_put(b, "null");
    }
}

/* dumps_calls: args already inserted in sorted key order (schema keys are
 * iterated sorted), so no re-sort is needed — but canon_args sorts anyway;
 * keys within one call were filled in sorted order by construction. */
static char *calls_to_json(const Call *calls, int n) {
    SB b;
    sb_init(&b);
    sb_put(&b, "[");
    for (int i = 0; i < n; i++) {
        if (i) sb_put(&b, ",");
        sb_put(&b, "{\"name\":\"");
        json_escape_to(&b, calls[i].name);
        sb_put(&b, "\",\"arguments\":{");
        for (int a = 0; a < calls[i].args->n; a++) {
            if (a) sb_put(&b, ",");
            sb_put(&b, "\"");
            json_escape_to(&b, calls[i].args->keys[a]);
            sb_put(&b, "\":");
            jval_dump(&b, calls[i].args->items[a]);
        }
        sb_put(&b, "}}");
    }
    sb_put(&b, "]");
    return b.buf;
}

/* ------------------------------------------------------ grammar decode */

static void args_add(JVal *args, const char *key, JVal *val) {
    args->keys = realloc(args->keys, sizeof(char *) * (args->n + 1));
    args->items = realloc(args->items, sizeof(JVal *) * (args->n + 1));
    args->keys[args->n] = xstrdup(key);
    args->items[args->n] = val;
    args->n++;
}

/* _fill_args: positioned just after `"arguments":{`, leaves after `}}` */
static Call fill_args(Dec *d, const Tool *tool) {
    Call call;
    call.name = xstrdup(tool->name);
    call.args = jnew(J_OBJ);
    int first = 1;
    for (int ki = 0; ki < tool->n_params; ki++) {
        const Param *p = &tool->params[ki];
        const char *sep = first ? "" : ",";
        char opener[4200];
        snprintf(opener, sizeof opener, "%s\"%s\":", sep, p->key);
        if (!p->required) {
            char skip[4200];
            if (ki + 1 < tool->n_params)
                snprintf(skip, sizeof skip, "%s\"%s\":", sep, tool->params[ki + 1].key);
            else
                snprintf(skip, sizeof skip, "}");
            const char *opts[2] = { opener, skip };
            if (dec_choose_str(d, opts, 2) != 0) continue;
        }
        dec_feed_str(d, opener);
        first = 0;
        if (p->enum_vals) {
            dec_feed_str(d, "\"");
            int ne = p->enum_vals->n;
            char **opts = xmalloc(sizeof(char *) * ne);
            for (int e = 0; e < ne; e++) opts[e] = jval_str(p->enum_vals->items[e]);
            int pick = dec_choose_str(d, (const char **)opts, ne);
            char fed[4200];
            snprintf(fed, sizeof fed, "%s\"", opts[pick]);
            dec_feed_str(d, fed);
            JVal *v = jnew(J_STR);
            v->s = xstrdup(opts[pick]);
            args_add(call.args, p->key, v);
            for (int e = 0; e < ne; e++) free(opts[e]);
            free(opts);
        } else if (strcmp(p->type, "boolean") == 0) {
            const char *opts[2] = { "true", "false" };
            int pick = dec_choose_first(d, opts, 2);
            dec_feed_str(d, opts[pick]);
            JVal *v = jnew(J_BOOL);
            v->b = (pick == 0);
            args_add(call.args, p->key, v);
        } else if (strcmp(p->type, "integer") == 0 || strcmp(p->type, "number") == 0) {
            char *s = dec_gen_number(d);
            double num = 0.0;
            int isint = 0;
            int ok = is_json_number(s, &num, &isint);
            if (!ok) {
                /* longest valid numeric prefix, else 0 */
                num = 0.0;
                isint = 1;
                for (size_t cut = strlen(s); cut > 0; cut--) {
                    char *pref = xstrndup(s, cut);
                    double n2;
                    int ii;
                    int good = is_json_number(pref, &n2, &ii);
                    free(pref);
                    if (good) { num = n2; isint = ii; ok = 1; break; }
                }
            }
            JVal *v;
            if (strcmp(p->type, "integer") == 0 && !isint) {
                v = jnew(J_INT);
                v->i = (long long)num;   /* Python int(): truncate toward zero */
            } else if (isint) {
                v = jnew(J_INT);
                v->i = (long long)num;
            } else {
                v = jnew(J_FLOAT);
                v->f = num;
            }
            free(s);
            args_add(call.args, p->key, v);
        } else {
            dec_feed_str(d, "\"");
            char *val = dec_gen_string(d);
            dec_feed_str(d, "\"");
            JVal *v = jnew(J_STR);
            v->s = val;
            args_add(call.args, p->key, v);
        }
    }
    dec_feed_str(d, "}}");
    return call;
}

/* constrained_decode (gated, name head on, k=0 default, temp=0) */
static int decode_calls(Model *m, Tok *t, const char *query,
                        const Tool *tools, int n_tools, Call *out_calls) {
    if (!n_tools) return 0;
    int k = n_tools <= 8 ? n_tools : 8;
    char *prompt = render_prompt(query, tools, n_tools);
    Dec *d = dec_new(m, t);
    int pids[8192];
    int n_pids = tok_encode(t, prompt, pids, 8192);
    SpanTok *spans = xmalloc(sizeof(SpanTok) * n_tools);
    name_spans(t, prompt, pids, n_pids, tools, n_tools, spans);

    int bos = BOS;
    dec_feed_ids(d, &bos, 1);
    dec_feed_str(d, prompt);
    dec_feed_str(d, "[");

    int n_emitted = 0;
    int stopped_via_bracket = 0;
    for (int budget = 0; budget < MAX_CALLS; budget++) {
        const char *open_opt = n_emitted == 0 ? "{\"name\":\"" : ",{\"name\":\"";
        const char *stop_opts[2];
        stop_opts[0] = "]";
        stop_opts[1] = open_opt;
        double stop_lp[2];
        dec_score_first(d, stop_opts, 2, stop_lp);
        int refuse = stop_lp[0] >= stop_lp[1];   /* argmax; index 0 wins ties */
        if (refuse && n_emitted == 0) {
            /* gated refusal override */
            if (n_tools > 1) {
                int all_idx[512];
                for (int i = 0; i < n_tools && i < 512; i++) all_idx[i] = i;
                double *lex = xmalloc(sizeof(double) * n_tools);
                lexical_scores_c(query, tools, all_idx, n_tools, NULL, 0, lex);
                double mx = lex[0];
                for (int i = 1; i < n_tools; i++) if (lex[i] > mx) mx = lex[i];
                int denom = n_tools > 2 ? n_tools : 2;
                double peak = mx - 1.0 / denom;
                double margin = fabs(stop_lp[0] - stop_lp[1]);
                if (peak > 0.25 || (margin < REFUSE_GATE && peak > 0.08)) refuse = 0;
                free(lex);
            }
        }
        if (refuse) {
            dec_feed_str(d, "]");
            stopped_via_bracket = 1;
            break;
        }
        dec_feed_str(d, open_opt);

        int cand_idx[512];
        int n_cand = retrieve_tools(query, tools, n_tools, k, out_calls, n_emitted, cand_idx);
        const char **cand_names = xmalloc(sizeof(char *) * n_cand);
        for (int i = 0; i < n_cand; i++) cand_names[i] = tools[cand_idx[i]].name;

        /* Python: use the head iff every CANDIDATE has a resolved span */
        int cand_spans_ok = 1;
        for (int i = 0; i < n_cand; i++)
            if (!spans[cand_idx[i]].found) cand_spans_ok = 0;

        double probs[512];
        if (cand_spans_ok) {
            /* head scores: bilinear h_dec . W . mean(h_span) */
            int dpos = d->n_ids - 1;
            float *dec_vec = xmalloc(sizeof(float) * m->d);
            mw_matvec(&m->name_head, d->hidden + (size_t)dpos * m->d, dec_vec);
            float *head = xmalloc(sizeof(float) * n_cand);
            for (int i = 0; i < n_cand; i++) {
                const SpanTok *sp = &spans[cand_idx[i]];
                float acc = 0.0f;
                int len = sp->end - sp->start;
                for (int jd = 0; jd < m->d; jd++) {
                    float mean = 0.0f;
                    for (int r = sp->start; r < sp->end; r++)
                        mean += d->hidden[(size_t)r * m->d + jd];
                    acc += (mean / len) * dec_vec[jd];
                }
                head[i] = acc;
            }
            softmax_(head, n_cand);
            double lm[512];
            dec_score_str(d, cand_names, n_cand, lm);
            /* softmax over lm */
            double mx = lm[0];
            for (int i = 1; i < n_cand; i++) if (lm[i] > mx) mx = lm[i];
            double sum = 0.0;
            for (int i = 0; i < n_cand; i++) { lm[i] = exp(lm[i] - mx); sum += lm[i]; }
            for (int i = 0; i < n_cand; i++) probs[i] = (head[i] + lm[i] / sum) / 2.0;
            free(head);
            free(dec_vec);
        } else {
            double lm[512];
            dec_score_str(d, cand_names, n_cand, lm);
            double mx = lm[0];
            for (int i = 1; i < n_cand; i++) if (lm[i] > mx) mx = lm[i];
            double sum = 0.0;
            for (int i = 0; i < n_cand; i++) { lm[i] = exp(lm[i] - mx); sum += lm[i]; }
            for (int i = 0; i < n_cand; i++) probs[i] = lm[i] / sum;
        }
        if (n_cand > 1) {
            double lex[512];
            lexical_scores_c(query, tools, cand_idx, n_cand, out_calls, n_emitted, lex);
            double lsum = 0.0;
            for (int i = 0; i < n_cand; i++) lsum += lex[i];
            if (lsum > 0) {
                for (int i = 0; i < n_cand; i++) lex[i] /= lsum;
                double uniform = 1.0 / n_cand;
                double lmax = lex[0];
                for (int i = 1; i < n_cand; i++) if (lex[i] > lmax) lmax = lex[i];
                double peak = lmax - uniform;
                /* confidence: top1 - top2 of probs */
                double p1 = -1, p2 = -1;
                for (int i = 0; i < n_cand; i++) {
                    if (probs[i] > p1) { p2 = p1; p1 = probs[i]; }
                    else if (probs[i] > p2) p2 = probs[i];
                }
                double confidence = p1 - p2;
                double w = peak * LEX_SHARPNESS;
                if (w < 0) w = 0;
                if (w > LEX_MAX_WEIGHT) w = LEX_MAX_WEIGHT;
                w *= (1.0 - confidence);
                for (int i = 0; i < n_cand; i++)
                    probs[i] = (1.0 - w) * probs[i] + w * lex[i];
            }
        }
        int pick = 0;
        for (int i = 1; i < n_cand; i++) if (probs[i] > probs[pick]) pick = i;
        const Tool *tool = &tools[cand_idx[pick]];
        dec_feed_str(d, tool->name);
        dec_feed_str(d, "\",\"arguments\":{");
        out_calls[n_emitted++] = fill_args(d, tool);
        free(cand_names);
    }
    if (!stopped_via_bracket) dec_feed_str(d, "]");   /* for-else in Python */
    free(spans);
    free(prompt);
    dec_free(d);
    return n_emitted;
}

/* --------------------------------------------------------- wasm API */

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>

static Model *g_model = NULL;
static Tok *g_tok = NULL;

EMSCRIPTEN_KEEPALIVE
int th_init(const char *wpath, const char *tpath) {
    g_model = model_load(wpath);
    g_tok = tok_load(tpath);
    return (g_model && g_tok) ? 0 : 1;
}

/* One request: catalog JSON + query in, dumps_calls JSON out.
 * Returned string is owned by the engine and valid until the next call.
 * The page pre-validates the catalog JSON, so die() paths are unreachable
 * from well-formed UI input. */
EMSCRIPTEN_KEEPALIVE
const char *th_call(const char *catalog_json, const char *query) {
    static char *last = NULL;
    free(last);
    last = NULL;
    if (!g_model || !g_tok) return "{\"error\":\"engine not initialized\"}";
    int n_tools = 0;
    Tool *tools = tools_from_json(jparse(catalog_json), &n_tools);
    Call calls[MAX_CALLS];
    double t0 = now_ms();
    int n = decode_calls(g_model, g_tok, query, tools, n_tools, calls);
    double dt = now_ms() - t0;
    char *body = calls_to_json(calls, n);
    SB out;
    sb_init(&out);
    char head[64];
    snprintf(head, sizeof head, "{\"ms\":%.0f,\"calls\":", dt);
    sb_put(&out, head);
    sb_put(&out, body);
    sb_put(&out, "}");
    free(body);
    last = out.buf;
    return last;
}
#endif

/* ------------------------------------------------------------- main */

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open input file");
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = xmalloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("read error");
    buf[n] = 0;
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    const char *wpath = NULL, *tpath = NULL, *cpath = NULL, *jsonl = NULL;
    const char *query = NULL;
    int stats = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) wpath = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) tpath = argv[++i];
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) cpath = argv[++i];
        else if (strcmp(argv[i], "--jsonl") == 0 && i + 1 < argc) jsonl = argv[++i];
        else if (strcmp(argv[i], "--stats") == 0) stats = 1;
        else query = argv[i];
    }
    if (!wpath || !tpath || (!jsonl && (!cpath || !query))) {
        fprintf(stderr,
            "usage: thimble -w weights.bin -t tokenizer.bin -c catalog.json \"query\"\n"
            "       thimble -w weights.bin -t tokenizer.bin --jsonl rows.jsonl\n");
        return 2;
    }
    double t0 = now_ms();
    Model *m = model_load(wpath);
    Tok *t = tok_load(tpath);
    double t_load = now_ms() - t0;

    double t1 = now_ms();
    int n_rows = 0;
    if (jsonl) {
        char *all = read_file(jsonl);
        char *line = strtok(all, "\n");
        while (line) {
            if (*line) {
                JVal *row = jparse(line);
                const char *q = jstr(jget(row, "query"), "");
                int n_tools = 0;
                Tool *tools = tools_from_json(jget(row, "tools"), &n_tools);
                Call calls[MAX_CALLS];
                int n = decode_calls(m, t, q, tools, n_tools, calls);
                char *out = calls_to_json(calls, n);
                puts(out);
                fflush(stdout);
                free(out);
                n_rows++;
            }
            line = strtok(NULL, "\n");
        }
    } else {
        char *cat = read_file(cpath);
        int n_tools = 0;
        Tool *tools = tools_from_json(jparse(cat), &n_tools);
        Call calls[MAX_CALLS];
        int n = decode_calls(m, t, query, tools, n_tools, calls);
        char *out = calls_to_json(calls, n);
        puts(out);
        free(out);
        n_rows = 1;
    }
    if (stats) {
        double dt = now_ms() - t1;
        fprintf(stderr,
            "load: %.0f ms | rows: %d | forwards: %ld | decode: %.0f ms "
            "(%.1f ms/row, %.0f fwd/s)\n",
            t_load, n_rows, g_forwards, dt,
            n_rows ? dt / n_rows : 0.0, dt > 0 ? g_forwards * 1000.0 / dt : 0.0);
    }
    return 0;
}
