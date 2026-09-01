"""Generate a reproducible 5,000-row synthetic support dataset for pipeline validation."""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.domain.taxonomy import CATEGORIES


TEMPLATES = {
    "account_access": ["reset my password", "account is locked", "cannot sign in", "login is failing", "locked out after failed login attempts and need my corporate account unlocked"],
    "email": ["email is not sending", "mailbox is full", "calendar is not syncing", "messages are delayed"],
    "network": ["VPN will not connect", "office Wi-Fi is down", "network connection drops", "cannot reach the LAN", "cannot access internal applications from home", "company VPN disconnects every few minutes"],
    "hardware": ["laptop will not start", "monitor is blank", "keyboard is broken", "desktop has no power"],
    "software_installation": ["need approved software installed", "application update failed", "please install the tool", "software setup is needed"],
    "application_error": ["business application shows an error", "app crashes on launch", "system returns an exception", "application is failing"],
    "printer": ["printer will not print", "scanner is unavailable", "print queue is stuck", "cannot connect to printer"],
    "security": ["received a phishing email", "suspect malware on device", "security alert appeared", "need to report a suspicious message", "suspicious email asks me to verify my password using an external link", "suspicious link may be trying to steal credentials"],
    "access_request": ["request access to a shared drive", "need permissions for the application", "please grant role access", "need access to the folder"],
    "data_recovery": ["deleted files need recovery", "missing data from my folder", "restore an old document", "lost files after an incident"],
    "performance": ["computer is running slowly", "application performance is poor", "device freezes frequently", "system response is very slow", "laptop takes ten minutes to start and applications freeze", "workstation is slow after startup"],
    "telephony": ["desk phone has no dial tone", "softphone microphone fails", "cannot make a call", "voice service is down"],
    "virtual_desktop": ["remote desktop will not open", "VDI session disconnects", "virtual workspace is unavailable", "remote desktop is frozen"],
    "onboarding": ["new starter needs a laptop", "provision equipment for a new joiner", "onboarding account setup is needed", "prepare access for a new employee"],
    "offboarding": ["leaver access must be removed", "recover equipment from departing employee", "offboarding account closure is needed", "disable a former user's access"],
    "how_to": ["how do I change this setting", "please explain how to use the portal", "need guidance on this process", "where can I find the instructions"],
    "other": ["I have an IT issue", "please review this support request", "need general technical help", "this request does not fit another category"],
}


def generate(output: Path, rows: int, seed: int) -> None:
    random.seed(seed)
    labels = [category.name for category in CATEGORIES]
    records = []
    for index in range(rows):
        label = labels[index % len(labels)]
        template = random.choice(TEMPLATES[label])
        urgency = random.choice(["today", "as soon as possible", "this morning", "during work"])
        context = random.choice(["for my team", "at the office", "on my work device", "after the latest change"])
        records.append({
            "case_id": f"SYN-{index + 1:05d}",
            "text": f"{template} {context}; please resolve {urgency}. Reference {index + 1:05d}.",
            "label": label,
        })
    random.shuffle(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"Wrote {len(records)} synthetic records to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/cases_5000.csv"))
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.rows, args.seed)
