# Security policy

## Reporting a vulnerability

If you find a security issue in Thimble — the Python pipeline, the C engine,
or the published weights — please report it privately via
[GitHub security advisories](https://github.com/nikshepsvn/thimble/security/advisories/new)
or email nikshepsvn@gmail.com. Please do not open a public issue for
exploitable problems.

You can expect an acknowledgement within a few days. There is no bug bounty;
this is a solo open-source project.

## Scope notes

- The C engine (`cengine/thimble.c`) parses untrusted JSON catalogs and
  queries; memory-safety reports there are especially welcome.
- The model itself emits schema-constrained JSON only; treat its *content*
  (argument values) as untrusted user-derived data downstream, as you would
  any model output.
