"""Device-use + generic API catalog with train / ood split and Hammer-style aliases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict
    required: list[str]
    split: str = "train"
    domain: str = "home"


def schema(t: Tool) -> dict:
    return {
        "name": t.name,
        "description": t.description,
        "parameters": {"type": "object", "properties": t.properties, "required": t.required},
    }


ROOMS = ["kitchen", "living room", "bedroom", "bathroom", "office", "garage", "hallway", "basement"]
CITIES = ["Lagos", "Austin", "Tokyo", "Berlin", "Nairobi", "Seoul", "Lisbon", "Oslo"]
NAMES = ["Maya Chen", "Omar Diallo", "Priya Shah", "Leo Park", "Ana Ruiz"]
BRIGHT = [0, 10, 20, 30, 50, 70, 80, 100]
TEMPS = [16, 18, 20, 21, 22, 24, 26]
HANDLES = ["@maya", "@omar", "@priya", "@leo"]

TRAIN: list[Tool] = [
    Tool("set_lights", "Set a room's lights on/off and brightness.", {
        "room": {"type": "string"}, "on": {"type": "boolean"},
        "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
    }, ["room"], "train", "home"),
    Tool("set_thermostat", "Set target temperature and mode.", {
        "temperature": {"type": "integer"}, "mode": {"type": "string", "enum": ["heat", "cool", "auto"]},
    }, ["temperature"], "train", "home"),
    Tool("lock_door", "Lock or unlock a named door.", {
        "door": {"type": "string"}, "locked": {"type": "boolean"},
    }, ["door", "locked"], "train", "home"),
    Tool("play_music", "Play music by query.", {"query": {"type": "string"}}, ["query"], "train", "media"),
    Tool("set_volume", "Set speaker volume 0-100.", {"level": {"type": "integer"}}, ["level"], "train", "media"),
    Tool("create_calendar_event", "Create a calendar event.", {
        "title": {"type": "string"}, "datetime": {"type": "string"}, "duration_min": {"type": "integer"},
    }, ["title", "datetime"], "train", "phone"),
    Tool("set_timer", "Start a timer in minutes.", {"minutes": {"type": "integer"}}, ["minutes"], "train", "phone"),
    Tool("send_message", "Text a contact.", {
        "to": {"type": "string"}, "body": {"type": "string"},
    }, ["to", "body"], "train", "phone"),
    Tool("show_map", "Show a place on the map.", {"place": {"type": "string"}}, ["place"], "train", "phone"),
    Tool("get_weather", "Current weather for a city.", {"city": {"type": "string"}}, ["city"], "train", "api"),
    Tool("turn_on_flashlight", "Turn the flashlight on.", {}, [], "train", "phone"),
    Tool("turn_off_flashlight", "Turn the flashlight off.", {}, [], "train", "phone"),
    Tool("open_wifi_settings", "Open Wi-Fi settings.", {}, [], "train", "phone"),
    Tool("create_contact", "Add a contact.", {
        "first_name": {"type": "string"}, "last_name": {"type": "string"}, "phone_number": {"type": "string"},
    }, ["first_name"], "train", "phone"),
    Tool("send_email", "Send an email.", {
        "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"},
    }, ["to", "subject"], "train", "phone"),
    Tool("robot_move", "Move a robot by meters and heading.", {
        "meters": {"type": "number"}, "heading": {"type": "string", "enum": ["forward", "back", "left", "right"]},
    }, ["meters", "heading"], "train", "robot"),
    Tool("extract_invoice", "Extract invoice fields.", {
        "vendor": {"type": "string"}, "total": {"type": "number"}, "due_date": {"type": "string"},
    }, ["vendor", "total"], "train", "extract"),
]

OOD: list[Tool] = [
    Tool("adjust_lamp", "Adjust a lamp in a room.", {
        "room": {"type": "string"}, "on": {"type": "boolean"},
        "brightness": {"type": "integer"},
    }, ["room"], "ood", "home"),
    Tool("set_climate", "Set climate control.", {
        "temperature": {"type": "integer"}, "mode": {"type": "string"},
    }, ["temperature"], "ood", "home"),
    Tool("add_agenda_item", "Add an agenda item.", {
        "title": {"type": "string"}, "datetime": {"type": "string"},
    }, ["title", "datetime"], "ood", "phone"),
    Tool("compose_message", "Compose a text.", {
        "to": {"type": "string"}, "body": {"type": "string"},
    }, ["to", "body"], "ood", "phone"),
    Tool("lookup_forecast", "Look up a city forecast.", {"city": {"type": "string"}}, ["city"], "ood", "api"),
]


def by_split(split: str) -> list[Tool]:
    return [t for t in TRAIN + OOD if t.split == split]


def by_name(name: str) -> Tool:
    for t in TRAIN + OOD:
        if t.name == name:
            return t
    raise KeyError(name)
