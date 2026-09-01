from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    description: str


TAXONOMY_VERSION = "2026.09.0"
CATEGORIES = (
    Category(1, "account_access", "Account lockout, password, or sign-in issues"),
    Category(2, "email", "Email delivery, mailbox, or calendar issues"),
    Category(3, "network", "LAN, Wi-Fi, VPN, or connectivity issues"),
    Category(4, "hardware", "Laptop, desktop, monitor, or peripheral issues"),
    Category(5, "software_installation", "Installing or updating approved software"),
    Category(6, "application_error", "Errors or failures in business applications"),
    Category(7, "printer", "Printing, scanning, or printer connectivity"),
    Category(8, "security", "Phishing, malware, or security policy concerns"),
    Category(9, "access_request", "Request for access, role, or permissions"),
    Category(10, "data_recovery", "Deleted, missing, or unrecoverable data"),
    Category(11, "performance", "Slow device, application, or system performance"),
    Category(12, "telephony", "Desk phone, softphone, or voice issues"),
    Category(13, "virtual_desktop", "VDI, remote desktop, or virtual workspace issues"),
    Category(14, "onboarding", "New starter setup and equipment provisioning"),
    Category(15, "offboarding", "Leaver access removal and asset return"),
    Category(16, "how_to", "How-to questions and general guidance"),
    Category(17, "other", "Unclassified or unsupported support requests"),
)
CATEGORY_BY_NAME = {category.name: category for category in CATEGORIES}


def validate_category(name: str) -> str:
    if name not in CATEGORY_BY_NAME:
        raise ValueError(f"Unknown taxonomy category: {name}")
    return name
