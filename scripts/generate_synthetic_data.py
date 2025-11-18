import csv
import random
from pathlib import Path

OUTPUT_PATH = Path("data/transactions_dataset.csv")

CATEGORIES = [
    "Dining",
    "Groceries",
    "Shopping",
    "Fuel",
    "Bills",
    "Travel",
    "Entertainment",
    "Others",
]

AMOUNT_BANDS = ["0-500", "500-2000", "2000-5000", "5000+"]
REGIONS = ["IN", "US", "EU", "SG"]

MERCHANTS_BY_CAT = {
    "Dining": [
        "Starbucks",
        "McDonald's",
        "Domino's Pizza",
        "KFC",
        "Subway",
        "Cafe Coffee Day",
    ],
    "Groceries": [
        "Big Bazaar",
        "DMart",
        "Reliance Fresh",
        "Walmart",
        "Target Groceries",
    ],
    "Shopping": [
        "Amazon",
        "Flipkart",
        "Myntra",
        "Ajio",
        "Best Buy",
    ],
    "Fuel": [
        "Shell Petrol",
        "HP Petrol Pump",
        "IndianOil",
        "BP Fuel Station",
    ],
    "Bills": [
        "Airtel Postpaid",
        "Jio Fiber",
        "Electricity Board",
        "Water Utility",
        "Gas Company",
    ],
    "Travel": [
        "Uber",
        "Ola Cabs",
        "IndiGo Airlines",
        "MakeMyTrip",
        "IRCTC",
    ],
    "Entertainment": [
        "Netflix",
        "Spotify",
        "BookMyShow",
        "PVR Cinemas",
        "SonyLiv",
    ],
    "Others": [
        "Service Charge",
        "Bank Fee",
        "Maintenance Charge",
        "Parking",
        "Toll Plaza",
    ],
}


DECORATIONS = [
    "",
    " #0421",
    " *REF 9982",
    " - ONLINE",
    " - POS TXN",
    " INTL",
    " MUMBAI IN",
    " BLR IN",
    " NEW DELHI",
]


def make_raw(merchant: str) -> str:
    return merchant + random.choice(DECORATIONS)


def normalise(text: str) -> str:
    cleaned = (
        text.lower()
        .replace("*", "")
        .replace("-", " ")
        .replace("#", "")
    )
    return " ".join(cleaned.split())


def main(n_samples: int = 100):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, n_samples + 1):
        category = random.choice(CATEGORIES)
        merchant = random.choice(MERCHANTS_BY_CAT[category])
        raw = make_raw(merchant)
        norm = normalise(raw)
        band = random.choice(AMOUNT_BANDS)
        region = random.choice(REGIONS)

        rows.append(
            {
                "transaction_id": f"T{i:04d}",
                "raw_description": raw,
                "normalised_description": norm,
                "category": category,
                "amount_band": band,
                "region": region,
            }
        )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "transaction_id",
                "raw_description",
                "normalised_description",
                "category",
                "amount_band",
                "region",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Wrote {len(rows)} synthetic rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
