"""The rental a result was measured on, as the person running gpu-mark states it.

gpu-mark doesn't detect any of this: a provider's metadata service differs per
provider and some have none. A field not given is recorded as null and listed
under ``not_given``, so a result never carries a guessed price or region.
"""

from __future__ import annotations

import argparse
from datetime import datetime

FIELDS = ("instance_id", "region", "price_usd_per_hour", "price_source", "rental_started_at")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--instance-id", help="the provider's id for the rented machine or pod")
    p.add_argument("--region", help="the provider's region or data center name")
    p.add_argument("--price-usd-per-hour", type=float, help="the hourly price paid for this machine, in US dollars")
    p.add_argument("--price-source", help="where the price came from, e.g. a console page or an invoice line")
    p.add_argument("--rental-started-at", help="when the rental began, ISO 8601 with a UTC offset")


def from_args(args: argparse.Namespace) -> dict:
    values = {f: getattr(args, f) for f in FIELDS}
    if values["price_usd_per_hour"] is not None and not values["price_source"]:
        raise SystemExit("--price-usd-per-hour needs --price-source: a price without its source can't be checked")
    if values["price_usd_per_hour"] is not None and values["price_usd_per_hour"] <= 0:
        raise SystemExit(f"--price-usd-per-hour must be positive, got {values['price_usd_per_hour']}")
    if values["rental_started_at"] is not None:
        t = datetime.fromisoformat(values["rental_started_at"])
        if t.tzinfo is None:
            raise SystemExit("--rental-started-at needs a UTC offset, e.g. 2026-11-16T18:00:00+00:00")
    return {**values, "not_given": [f for f in FIELDS if values[f] is None]}
