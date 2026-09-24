"""Write a synthetic spam/ham corpus and a small sample mailbox.

Provenance: every message is original text generated for this educational
project. The file does not include Enron mail, SpamAssassin public corpora,
Ling-Spam, or the SMS Spam Collection.

Run from the repo root:

    python scripts/generate_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import mailbox
import random
import sys
from collections import Counter
from email.message import EmailMessage
from pathlib import Path
from string import Formatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 42
PER_LABEL = 800

FIRST_NAMES = [
    "Amina", "Priya", "Elena", "Marcus", "Jonah", "Sofia", "Wei", "Noah",
    "Lila", "Andre", "Grace", "Omar", "Hannah", "Diego", "Ruth", "Kenji",
]
LAST_NAMES = [
    "Nair", "Okoye", "Berg", "Silva", "Chen", "Ibarra", "Park", "Adler",
    "Mensah", "Novak", "Shah", "Duarte", "Klein", "Rossi", "Abebe", "Frost",
]
COMPANIES = [
    "Northwind", "Brightline Studio", "Lumen College", "Harbor Credit Union",
    "Metro Transit", "Fieldnote Labs", "Copperline Health", "Aster Library",
]
PRODUCTS = [
    "mailbox", "VPN", "payroll portal", "benefits account", "cloud drive",
    "student portal", "wire desk", "badge access",
]
DEPTS = ["Security", "IT", "People Ops", "Finance", "Facilities", "Registrar", "Support"]
TEAMS = ["platform", "detection", "accounts payable", "student success", "infrastructure", "clinic ops"]
VENDORS = ["Paperlane Supplies", "Northstar Catering", "Helio Print", "Cedar Hosting"]
BANKS = ["Harbor Credit Union", "First Orchard Bank", "Lane and Mercer", "Civic Savings"]
COURSES = ["Network Defense", "Applied Machine Learning", "Incident Response", "Secure Coding"]
BRANDS = ["PayPal", "Microsoft", "Amazon", "Apple", "Netflix", "LinkedIn", "DHL", "FedEx"]
AMOUNTS = ["250", "500", "1200", "2500", "4800", "7500", "12500", "25000"]
HOURS = ["2", "4", "6", "12", "24", "48"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["09:00", "10:30", "13:00", "15:00", "16:30"]
CITIES = ["Austin", "Seattle", "Chicago", "Denver", "Boston", "Atlanta", "Phoenix", "Minneapolis"]
FILENAMES = [
    "Invoice_copy.docm", "payment_update.zip", "voice_message.exe",
    "remit_form.js", "Q3_statement.xlsm", "shipping_label.htm",
]
PHISH_HOSTS = [
    "secure-login-update.example", "account-verify-support.example",
    "paypal-resolution.example", "microsoft365-reset.example",
    "parcel-hold-notice.example", "wallet-sync-now.example",
    "invoice-remit-portal.example", "tax-refund-claim.example",
    "appleid-unlock.example", "dhl-hold-release.example",
]
BRAND_HOSTS = {
    "PayPal": "paypal-resolution.example",
    "Microsoft": "microsoft365-reset.example",
    "Amazon": "account-verify-support.example",
    "Apple": "appleid-unlock.example",
    "Netflix": "secure-login-update.example",
    "LinkedIn": "account-verify-support.example",
    "DHL": "dhl-hold-release.example",
    "FedEx": "parcel-hold-notice.example",
}
BENIGN_HOSTS = [
    "intranet.northwind.example", "docs.northwind.example",
    "calendar.northwind.example", "drive.brightline.example", "learn.lumen.example",
]
URL_PATHS = ["verify", "login", "reset", "secure", "update", "review", "notes", "agenda", "share/doc"]


def _family(label, category, subjects, intros, details, asks, closes):
    return {
        "label": label,
        "category": category,
        "subjects": subjects,
        "intros": intros,
        "details": details,
        "asks": asks,
        "closes": closes,
    }


FAMILIES = [
    _family(
        "spam",
        "credential_phish",
        [
            "{brand} password expires in {hours} hours",
            "Unusual sign-in to your {product}",
            "Confirm the {brand} login from {city}",
            "Your {company} mailbox will be locked",
        ],
        [
            "Dear {first},",
            "Hello {full},",
            "{brand} security alert for the {product} on file.",
            "A password reset was started for {address}.",
        ],
        [
            "The session started in {city} and used a browser we do not recognize.",
            "Someone requested a reset for the {brand} profile linked to your {product}.",
            "A forwarding rule was added to the mailbox owned by {full}.",
            "Directory records show the {product} is past due for verification.",
        ],
        [
            "Confirm the password within {hours} hours at {url}.",
            "Open {url} and verify the account before the hold is applied.",
            "Use the secure form at {url} to keep the mailbox active.",
            "Enter the current password at {url} so we can lift the lock.",
        ],
        [
            "If you ignore this note, access to {product} ends today.",
            "{brand} Account Services",
            "Reference {ticket}. This link expires in {hours} hours.",
            "Failure to verify will delete unread mail.",
        ],
    ),
    _family(
        "spam",
        "account_suspension",
        [
            "Account suspension notice {ticket}",
            "Final warning before we close {product}",
            "{company} access disabled in {hours} hours",
            "Immediate review required for {full}",
        ],
        [
            "{first}, your profile is on hold.",
            "We could not bill the card saved on the {product}.",
            "Automated notice: {address} is scheduled for suspension.",
            "Dear customer, the {dept} queue flagged your profile.",
        ],
        [
            "The hold started after a failed check from {city}.",
            "Storage for {product} exceeds the unpaid plan tied to {brand}.",
            "A compliance review listed {ticket} against {company}.",
            "Outgoing mail is already queued and will bounce after {hours} hours.",
        ],
        [
            "Restore service at {url} before the timer ends.",
            "Submit a copy of your password at {url} to reopen the profile.",
            "Avoid suspension by confirming ownership here: {url}",
            "Pay the balance and confirm the login at {url}.",
        ],
        [
            "Support will not accept a phone call for case {ticket}.",
            "This is the last notice we will send.",
            "{company} Billing Recovery",
            "Act now. Waiting until {day} is too late.",
        ],
    ),
    _family(
        "spam",
        "prize_lottery",
        [
            "You won {amount} USD in the {city} draw",
            "Prize desk: claim {amount} today",
            "Congratulations {first}, your ticket {ticket} hit",
            "Unclaimed reward expiring in {hours} hours",
        ],
        [
            "Dear winner {full},",
            "The {city} promotional desk selected {address}.",
            "Congratulations. A cash award of {amount} USD is waiting.",
            "Hello {first}, we have tried to reach you about a prize.",
        ],
        [
            "The drawing used ticket {ticket} and a sponsor named {brand}.",
            "Taxes and a release fee of {amount} USD must clear before the wire.",
            "A courier from {city} can deliver the check once you confirm identity.",
            "The prize pool was funded by {company} marketing and is still open.",
        ],
        [
            "Claim the funds at {url} with your password and date of birth.",
            "Reply with a copy of your ID and then open {url}.",
            "Pay the release fee through {url} to unlock the transfer.",
            "Send the name of your bank, {bank}, and complete {url}.",
        ],
        [
            "Unclaimed prizes return to the pool after {hours} hours.",
            "Prize desk {ticket}",
            "Do not tell your bank until the release fee posts.",
            "We congratulate you again, {first}.",
        ],
    ),
    _family(
        "spam",
        "advance_fee",
        [
            "Confidential transfer for {full}",
            "Estate funds of {amount} USD need a partner",
            "Can you receive a payment from {city}",
            "Private bank introduction {ticket}",
        ],
        [
            "Dear {first} {last},",
            "I am writing with a confidential business proposal.",
            "My late client left {amount} USD and no local heir.",
            "Greetings from {city}. I was given your email, {address}.",
        ],
        [
            "The funds sit at {bank} under reference {ticket}.",
            "I need a foreign partner to receive {amount} USD and keep ten percent.",
            "A lawyer in {city} will not release the file without a processing fee.",
            "{company} cannot be named on the wire, so the transfer must look personal.",
        ],
        [
            "Reply with your bank name and account officer.",
            "If you can help, open {url} and send the fee by card.",
            "Send a phone number and a copy of your passport.",
            "Confirm you will keep this private, then visit {url}.",
        ],
        [
            "I will travel to {city} once the first transfer lands.",
            "May God bless you for your cooperation.",
            "Delete this message after you reply.",
            "Your commission is {amount} USD.",
        ],
    ),
    _family(
        "spam",
        "fake_invoice",
        [
            "Invoice {ticket} is past due",
            "Payment failed for {vendor}",
            "Updated remittance advice",
            "{company} vendor bill for {amount} USD",
        ],
        [
            "Accounts payable,",
            "Hello {first}, the {dept} ledger is short.",
            "{vendor} rejected the last payment of {amount} USD.",
            "Please process invoice {ticket} today.",
        ],
        [
            "The purchase order covers {product} for the {team} team.",
            "A late fee starts in {hours} hours if the balance stays open.",
            "Our records show {bank} returned the ACH from {city}.",
            "The new bank details are attached and also posted at the portal.",
        ],
        [
            "Pay immediately at {url}.",
            "Open Attachment: {filename} and enable editing to see the account number.",
            "Wire {amount} USD today and upload the receipt at {url}.",
            "Change the vendor destination, then confirm at {url}.",
        ],
        [
            "{vendor} Collections",
            "Case {ticket} will go to a collection partner on {day}.",
            "Do not use the old account ending in {amount}.",
            "Thank you for settling this today.",
        ],
    ),
    _family(
        "spam",
        "malware_lure",
        [
            "Voice message from {coworker}",
            "Shared file: {filename}",
            "Scanned document for {dept}",
            "Re: contract draft {ticket}",
        ],
        [
            "Hi {first},",
            "{coworker} left a recording for you.",
            "The signed copy for {company} is ready.",
            "Hello, {dept} asked me to send this before {time}.",
        ],
        [
            "The player will not open in the browser, so the file is attached.",
            "Macros must be enabled or the totals for {amount} USD stay blank.",
            "I compressed the folder because the firewall in {city} blocked the link.",
            "The password for the archive is the ticket number {ticket}.",
        ],
        [
            "Download {filename} from {url}.",
            "Attachment: {filename}",
            "Open the file from {url} before the {day} meeting.",
            "Run the viewer and log in with your mailbox password.",
        ],
        [
            "Let me know if the audio is hard to hear.",
            "Thanks, {coworker}",
            "Sent from my phone in {city}.",
            "Please do this before {time}.",
        ],
    ),
    _family(
        "spam",
        "crypto_invest",
        [
            "Double your {amount} USD stake",
            "{brand} wallet needs syncing",
            "Private allocation for {first}",
            "Account {ticket} is out of date",
        ],
        [
            "Hi {full},",
            "Your wallet has been selected for a bonus round.",
            "We detected a login to the {brand} wallet from {city}.",
            "A strategist at {company} set aside a seat for {address}.",
        ],
        [
            "The pool returns {amount} USD for a {hours} hour lock.",
            "Unsynced wallets lose the promotional coins at midnight.",
            "A compliance hold will freeze trades unless the seed phrase is confirmed.",
            "The desk in {city} can mirror your position at {bank}.",
        ],
        [
            "Sync the wallet at {url}.",
            "Paste the recovery phrase into {url} to keep the allocation.",
            "Deposit through {url} before {day}.",
            "Reply to this email with the seed phrase if the site is down.",
        ],
        [
            "Seats close in {hours} hours.",
            "Wealth desk {ticket}",
            "Do not discuss the rate with support chat.",
            "Congratulations on the early access, {first}.",
        ],
    ),
    _family(
        "spam",
        "bec_wire",
        [
            "Need a wire before {time}",
            "Quick task from finance",
            "Do not call, just send {amount}",
            "Updated beneficiary for {vendor}",
        ],
        [
            "Hi {first}, are you at your desk?",
            "{coworker} is stuck in meetings and asked me to write.",
            "I need a quiet favor on the {team} account.",
            "Hello, this is time sensitive and I am about to board a flight.",
        ],
        [
            "Please wire {amount} USD to the new {vendor} beneficiary.",
            "The old {bank} instructions are wrong. Use the details I will send next.",
            "Legal in {city} will not close {ticket} until the transfer posts.",
            "I cannot talk on the phone. Email only until {day}.",
        ],
        [
            "Reply that you can send it in the next {hours} hours.",
            "Upload the confirmation to {url}.",
            "Use the account on {url} and mark the payment urgent.",
            "Send the wire, then tell only me. Skip the normal approver.",
        ],
        [
            "Thanks for handling this quietly.",
            "I owe you one.",
            "{full}",
            "Ping me when {bank} shows the debit.",
        ],
    ),
    _family(
        "spam",
        "shipping_phish",
        [
            "Package held in {city}",
            "{brand} delivery needs a fee",
            "Address problem for {full}",
            "Customs charge of {amount} USD",
        ],
        [
            "Hello {first},",
            "A parcel for {address} is waiting at the {city} depot.",
            "{brand} could not finish the delivery today.",
            "Dear customer, customs flagged shipment {ticket}.",
        ],
        [
            "The label is missing an apartment number and a duty of {amount} USD.",
            "The box returns to the sender after {hours} hours.",
            "The driver attempted delivery near {company} and no one signed.",
            "We need a card on file before the courier leaves {city}.",
        ],
        [
            "Release the package at {url}.",
            "Pay the duty and confirm your password at {url}.",
            "Print the new label from {url} and show it at the door.",
            "Update the delivery address here: {url}",
        ],
        [
            "{brand} Tracking",
            "Reference {ticket}.",
            "Storage fees start on {day}.",
            "This notice replaces the paper slip.",
        ],
    ),
    _family(
        "spam",
        "tech_support",
        [
            "We detected a virus on your PC",
            "Technician {ticket} is waiting",
            "Call about the {product} alert",
            "Remote support for {company}",
        ],
        [
            "Dear {full},",
            "A scan of {address} reported harmful software.",
            "This is technical support regarding your {product}.",
            "Hello {first}, the security subscription renewed incorrectly.",
        ],
        [
            "The error code {ticket} means someone in {city} can view your files.",
            "Your mailbox password is exposed until we clean the machine.",
            "A charge of {amount} USD was avoided, but the device is still infected.",
            "Ignore the warning and the {brand} account will be emptied.",
        ],
        [
            "Call the technician and install the tool from {url}.",
            "Download the cleaner at {url} and type your password when asked.",
            "Let us remote in. Start the session at {url}.",
            "Reply with a good time on {day} and the login for {product}.",
        ],
        [
            "Stay near the computer until the scan finishes.",
            "Support id {ticket}.",
            "Do not hire a local shop. They cannot see our ticket.",
            "We can waive the {amount} USD fee if you start today.",
        ],
    ),
    _family(
        "spam",
        "tax_refund",
        [
            "Refund of {amount} USD is pending",
            "Tax notice {ticket}",
            "Direct deposit could not post",
            "Action needed before {day}",
        ],
        [
            "Dear taxpayer {full},",
            "The revenue desk could not deposit funds to {bank}.",
            "Hello {first}, a refund is stalled for {address}.",
            "Official notice: case {ticket} needs a correction.",
        ],
        [
            "The refund amount is {amount} USD for the last filing year.",
            "The routing number on file was rejected in {city}.",
            "A penalty starts in {hours} hours if the profile stays incomplete.",
            "We compared the return with payroll data from {company}.",
        ],
        [
            "Submit bank details and your password at {url}.",
            "Claim the refund here: {url}",
            "Upload a photo of your ID through {url}.",
            "Confirm the {product} login so the payment can move.",
        ],
        [
            "Revenue notification {ticket}",
            "This mailbox is not monitored for questions.",
            "Unclaimed refunds expire on {day}.",
            "Do not forward this message.",
        ],
    ),
    _family(
        "spam",
        "romance_scam",
        [
            "I made it to {city}",
            "Can I ask a personal favor",
            "Thinking of you, {first}",
            "Stuck abroad, need {amount}",
        ],
        [
            "My dear {first},",
            "I hope this note finds you well.",
            "We have written for weeks, and I finally trust you.",
            "Hello, it is {coworker}. I am using a borrowed phone.",
        ],
        [
            "My bag was taken at the station in {city} and the embassy line is closed.",
            "The hotel wants {amount} USD before they will release my passport.",
            "I cannot reach {bank} from here, and the clerk will not take a promise.",
            "I am embarrassed to ask, but the flight home leaves on {day}.",
        ],
        [
            "Please wire {amount} USD and I will repay you on {day}.",
            "Send the money to the name I will give you, not to my old account.",
            "If a transfer is hard, pay the contact through {url}.",
            "Reply with a yes and I will send the details for {bank}.",
        ],
        [
            "I will explain everything when I land.",
            "Yours, {coworker}",
            "Please do not mention this to {company}.",
            "I am scared and I do not know who else to write.",
        ],
    ),
    _family(
        "ham",
        "meeting",
        [
            "Design review on {day}",
            "Can we move the {team} sync",
            "Agenda for {time}",
            "Notes from the {dept} standup",
        ],
        [
            "Hi {team} team,",
            "Hello {first},",
            "{coworker} asked me to send a time that works.",
            "Quick note ahead of {day}.",
        ],
        [
            "I would like to use the {time} slot to review the {product} notes.",
            "The open questions are already in the doc, so please read them beforehand.",
            "We only need {dept} and {team} in the room. Others can skip.",
            "{coworker} will walk through the decisions from the {city} offsite.",
        ],
        [
            "Reply with a yes or suggest another time on {day}.",
            "The agenda is on {url} for anyone already on the corporate network.",
            "Bring one risk and one decision. No slides are required.",
            "I will book the room near {dept} if I hear back today.",
        ],
        [
            "Thanks,\n{full}",
            "See you then.",
            "{first}",
            "I will send a recap after the meeting.",
        ],
    ),
    _family(
        "ham",
        "project_status",
        [
            "{product} status for {day}",
            "Where the {team} work landed",
            "Weekly update {ticket}",
            "Blockers before {time}",
        ],
        [
            "Team,",
            "Hi {first}, here is the short version.",
            "Status note for the {team} workstream.",
            "Sharing this so {dept} does not have to dig through chat.",
        ],
        [
            "The migration for {product} is on track for {day}.",
            "{coworker} finished the checklist, and I am reviewing the remaining gaps.",
            "We slipped the {vendor} integration because their sandbox was down in {city}.",
            "No customer data moved. This is still a staging exercise.",
        ],
        [
            "Read the board before standup if you have a dependency on {product}.",
            "Send blockers to {coworker}, not to the whole list.",
            "The working notes are at {url}.",
            "I will post the next update on {day} at {time}.",
        ],
        [
            "Thanks for the calm week.",
            "{full}\n{dept}",
            "Ping me if this summary missed your piece.",
            "Reference {ticket} if you open a task.",
        ],
    ),
    _family(
        "ham",
        "code_review",
        [
            "Review requested on the parser change",
            "Tests for the {team} branch",
            "Small fix before {day}",
            "Question about the {product} error path",
        ],
        [
            "Hi {coworker},",
            "Hello {first},",
            "I pushed a small change and I would like another set of eyes.",
            "{team} folks, this is ready for review.",
        ],
        [
            "The patch handles empty mailboxes without throwing.",
            "I kept the public function names stable for the {product} callers.",
            "Two unit tests cover the missing-file path and a normal message.",
            "There is no schema change and no new dependency.",
        ],
        [
            "Please leave comments by {time} on {day}.",
            "If it looks fine, merge it. I will watch the build.",
            "The diff is linked from {url}.",
            "I can walk through it after standup if that is easier.",
        ],
        [
            "Thanks,\n{full}",
            "No rush beyond {day}.",
            "I will rebase if main moves.",
            "{first}",
        ],
    ),
    _family(
        "ham",
        "academic",
        [
            "{course} reading for {day}",
            "Office hours with {coworker}",
            "Project checkpoint {ticket}",
            "Question about the lab write-up",
        ],
        [
            "Hi class,",
            "Hello {first},",
            "Students in {course},",
            "Hi {coworker}, I had a question after lecture.",
        ],
        [
            "The reading is the short section on evaluation, not the whole chapter.",
            "Bring a printed diagram if you want feedback on the {product} lab.",
            "Office hours stay at {time} in the {dept} room.",
            "Your checkpoint {ticket} only needs a paragraph on what you measured.",
        ],
        [
            "Submit the paragraph before {day}. Late notes are fine if you email first.",
            "The slides are posted at {url} for people with a campus login.",
            "Come to office hours or write {coworker} with a specific question.",
            "No extra software is required for this assignment.",
        ],
        [
            "Thank you,\n{full}",
            "See you in class.",
            "{dept}",
            "I will grade the checkpoints after {day}.",
        ],
    ),
    _family(
        "ham",
        "it_password_rotation",
        [
            "Lab passwords rotate on {day}",
            "Set your own {product} password",
            "No link in this note: credential rotation",
            "Maintenance window {day} {time}",
        ],
        [
            "Hi {team} team,",
            "Hello {first},",
            "This is a calendar notice from {dept}, not a login challenge.",
            "{company} will rotate {product} credentials on {day}.",
        ],
        [
            "You will choose the new password yourself on the portal already bookmarked on your laptop.",
            "We will not email a reset link, and nobody should ask you to reply with the current password.",
            "Shared lab accounts are listed in the internal runbook.",
            "If another message tells you to verify credentials before {day}, forward it to the security queue.",
        ],
        [
            "Finish the change before {time}. Bring questions to office hours with {coworker}.",
            "No reply is required if your account is outside the {team} rotation.",
            "The checklist is on {url} when you are on the corporate network.",
            "Add {day} to your calendar and ignore any other reset request.",
        ],
        [
            "We will never ask for your password.",
            "Thanks,\n{full}\n{dept}",
            "Reference {ticket} in the internal tracker.",
            "Questions can wait until standup.",
        ],
    ),
    _family(
        "ham",
        "security_awareness",
        [
            "Phishing drill recap",
            "How to report a suspicious message",
            "Examples from this month's lures",
            "Reminder: check the sender, then the ask",
        ],
        [
            "Hi everyone,",
            "Hello {team} team,",
            "Security awareness note for {day}.",
            "{first}, this is a recap, not a request to sign in.",
        ],
        [
            "The drill quoted fake package holds and fake password resets. Those links were inert examples.",
            "A real {dept} message will not ask you to type a password into a page we emailed.",
            "People who reported the sample did the right thing, even when the wording looked internal.",
            "The examples mentioned {brand} and {company} because those names are common bait.",
        ],
        [
            "Forward suspicious mail to the security queue and then delete it.",
            "Read the write-up on {url} if you want the screenshots.",
            "Do not click the links in a reported sample. The text in this note is enough.",
            "Tell {coworker} if a lure mentioned your team by name.",
        ],
        [
            "Thank you for the reports.",
            "{full}\n{dept}",
            "This mailbox accepts forwarded samples.",
            "No password, token, or code is required.",
        ],
    ),
    _family(
        "ham",
        "order_receipt",
        [
            "Receipt for the {vendor} order",
            "Your pickup is ready",
            "Order {ticket} confirmed",
            "Catering for {day}",
        ],
        [
            "Hi {first},",
            "Hello {full},",
            "Thanks for the order.",
            "{vendor} confirmation for {company}.",
        ],
        [
            "We charged {amount} USD to the card already on file with {dept}.",
            "The items will arrive at the {city} office on {day}.",
            "This receipt is for {product} supplies. It is not a new vendor setup.",
            "No bank details changed. {bank} will show the existing merchant name.",
        ],
        [
            "Reply if the quantity looks wrong.",
            "The invoice copy is on {url} for people with access to the drive.",
            "Bring the order number {ticket} if you pick the boxes up yourself.",
            "Nothing else is due. Please do not send a card number by email.",
        ],
        [
            "Thank you,\n{vendor}",
            "{full}",
            "We will deliver between {time} and the end of the day.",
            "Questions can go to {coworker}.",
        ],
    ),
    _family(
        "ham",
        "colleague",
        [
            "Got a minute on {day}",
            "Re: the {team} draft",
            "Thanks for covering standup",
            "Question about {product}",
        ],
        [
            "Hey {first},",
            "Hi {coworker},",
            "Hope the {city} trip was smooth.",
            "Quick question, nothing urgent.",
        ],
        [
            "I read the draft and I agree with the second option.",
            "The only part I would change is the example about {product}.",
            "I can take notes on {day} if you still need coverage.",
            "{dept} already approved the wording, so I would not reopen it.",
        ],
        [
            "Want to look at it together at {time}?",
            "Send me your edits whenever. Next week is fine.",
            "If you already answered this, ignore the duplicate.",
            "I left comments in the doc at {url}.",
        ],
        [
            "Thanks,\n{full}",
            "Appreciate it.",
            "{first}",
            "Talk soon.",
        ],
    ),
    _family(
        "ham",
        "calendar",
        [
            "Hold: {day} {time}",
            "Room change for the {team} review",
            "Canceled: office hours",
            "Invite update {ticket}",
        ],
        [
            "Hi {first},",
            "Calendar note only.",
            "Hello {team} team,",
            "{coworker} moved the hold.",
        ],
        [
            "The review is now {day} at {time}. The topic is still {product}.",
            "We lost the room near {dept}, so use the one by the library.",
            "Office hours on {day} are canceled because of the {city} workshop.",
            "The attachment on the invite is the same one-page agenda as last week.",
        ],
        [
            "Accept the update if the new time works.",
            "Decline if you are out. No project impact.",
            "The living agenda is at {url}.",
            "Reply only if the new room is a problem.",
        ],
        [
            "Thanks,\n{full}",
            "{dept}",
            "I will update the invite once more if {coworker} is still traveling.",
            "See you on {day}.",
        ],
    ),
    _family(
        "ham",
        "travel",
        [
            "Train to {city}",
            "Hotel confirmation {ticket}",
            "Out of office {day}",
            "Shared ride after {time}",
        ],
        [
            "Hi {first},",
            "Hello {coworker},",
            "Travel note for the {dept} visit.",
            "Hi team, I will be away.",
        ],
        [
            "I land in {city} on {day} around {time}.",
            "The hotel is already on the corporate card. Please do not send a card number.",
            "{coworker} and I will take the same train back.",
            "I am out {day} for the campus visit and will read mail at the end of the day.",
        ],
        [
            "The itinerary is at {url} if you need the address.",
            "Cover for {product} questions is {coworker}.",
            "No action if you were only copied for awareness.",
            "Meet in the lobby at {time} if you are on the same trip.",
        ],
        [
            "Thanks,\n{full}",
            "I will bring the notes back for {team}.",
            "Safe travels.",
            "{first}",
        ],
    ),
    _family(
        "ham",
        "soc_internal",
        [
            "Queue notes for {day}",
            "Case {ticket} closed",
            "Detection handoff at {time}",
            "Benign alert on {product}",
        ],
        [
            "Detection team,",
            "Hi {first},",
            "Handoff notes from the last shift.",
            "Internal case update. This is not a user notice.",
        ],
        [
            "Case {ticket} was a mistyped password loop from the {city} lab, not a takeover.",
            "We closed it after {coworker} confirmed the user was in the building.",
            "The {brand}-themed lure reported by {dept} matched this month's drill and was not delivered outside the sample list.",
            "No mailbox export was run. The next shift only needs to watch the {product} sensor.",
        ],
        [
            "Read the ticket before {time} if you are on call.",
            "Do not contact the user again. {coworker} already did.",
            "The timeline is in the case system, also linked from {url}.",
            "Page the lead only if a second account trips the same rule.",
        ],
        [
            "Thanks,\n{full}",
            "{dept} handoff",
            "Counts are in the shift channel.",
            "Nothing in this note asks anyone to reset a password.",
        ],
    ),
    _family(
        "ham",
        "personal",
        [
            "Dinner on {day}",
            "Can you feed the cat",
            "Article I mentioned",
            "Weekend plan",
        ],
        [
            "Hey {first},",
            "Hi,",
            "Hope your week is going okay.",
            "Hello {coworker},",
        ],
        [
            "I can do dinner on {day} after {time}.",
            "The soup place near the office is fine, or we can cook.",
            "I found the article about {course} and it is short.",
            "I will be in {city} visiting family and back on {day}.",
        ],
        [
            "Tell me if {day} is bad.",
            "No need to bring anything.",
            "The link is {url} if you want to read it on the train.",
            "I will call you tonight. No need to reply if you are busy.",
        ],
        [
            "{full}",
            "Talk later.",
            "Thanks for last week.",
            "See you soon.",
        ],
    ),
]


SAMPLE_MESSAGES = [
    {
        "from": "Priya Nair <priya.nair@northwind.example>",
        "to": "team@northwind.example",
        "subject": "Design review Thursday",
        "date": "Fri, 18 Sep 2026 14:00:00 +0000",
        "body": (
            "Hi team,\n\n"
            "Can we keep Thursday at 15:00 for the API design review? "
            "I will bring the open questions from last week's sync. "
            "The notes are already on the shared drive, so there is nothing to install "
            "and no password to send.\n\n"
            "Thanks,\nPriya"
        ),
    },
    {
        "from": "Security Queue <security@northwind.example>",
        "to": "all-staff@northwind.example",
        "subject": "How to report a suspicious message",
        "date": "Mon, 21 Sep 2026 15:10:00 +0000",
        "body": (
            "Hello everyone,\n\n"
            "This month's drill used fake package holds and fake password expirations. "
            "A real IT note will not ask you to type a password into a page that arrived by email. "
            "Forward the sample to the security queue and delete it. "
            "Do not click links inside a reported lure. This message is a recap only.\n\n"
            "Thank you,\nSecurity"
        ),
    },
    {
        "from": "Marcus Adler <marcus.adler@northwind.example>",
        "to": "platform@northwind.example",
        "subject": "Parser change ready for review",
        "date": "Tue, 22 Sep 2026 16:05:00 +0000",
        "body": (
            "Hi Jonah,\n\n"
            "I pushed a small change to the mailbox parser. Empty files no longer throw, "
            "and the public function names stayed the same. Two tests cover the missing-file path. "
            "Please leave comments by Thursday. No new dependency.\n\n"
            "Thanks,\nMarcus"
        ),
    },
    {
        "from": "IT Operations <it-ops@northwind.example>",
        "to": "lab-users@northwind.example",
        "subject": "Lab VPN passwords rotate Friday",
        "date": "Wed, 23 Sep 2026 13:30:00 +0000",
        "body": (
            "Hi lab users,\n\n"
            "VPN passwords rotate Friday at 18:00 UTC. You will set the new password yourself "
            "on the portal already bookmarked on your laptop. We will not email a reset link. "
            "If someone asks you to confirm the current password, forward that message to security. "
            "We will never ask for your password.\n\n"
            "IT Operations"
        ),
    },
    {
        "from": "Account Services <alerts@mailbox-verify-login.example>",
        "to": "priya.nair@northwind.example",
        "subject": "Mailbox locks in 2 hours",
        "date": "Wed, 23 Sep 2026 18:02:00 +0000",
        "body": (
            "Dear customer,\n\n"
            "We detected a sign-in from an unrecognized device. Verify your password now "
            "or the account will be suspended.\n\n"
            "http://mailbox-verify-login.example/secure\n\n"
            "This link expires in 2 hours. Failure to confirm will limit outgoing mail.\n\n"
            "Account Security Team"
        ),
    },
    {
        "from": "Prize Desk <claim@secure-login-update.example>",
        "to": "elena.berg@lumen.example",
        "subject": "You won 7500 USD in the Austin draw",
        "date": "Thu, 17 Sep 2026 11:22:00 +0000",
        "body": (
            "Dear winner Elena,\n\n"
            "A cash award of 7500 USD is waiting under ticket INC44021. "
            "Taxes and a release fee must clear before the wire. "
            "Claim the funds and enter your date of birth at "
            "http://secure-login-update.example/verify\n\n"
            "Unclaimed prizes return to the pool after 12 hours.\n"
        ),
    },
    {
        "from": "Collections <billing@invoice-remit-portal.example>",
        "to": "ap@northwind.example",
        "subject": "Invoice INC90210 is past due",
        "date": "Thu, 17 Sep 2026 08:14:00 +0000",
        "body": (
            "Accounts payable,\n\n"
            "Vendor bill 4800 USD is past due. A late fee starts in 6 hours. "
            "Open the remittance workbook and enable editing to see the new account number.\n\n"
            "Pay immediately at http://invoice-remit-portal.example/update\n\n"
            "Collections"
        ),
        "attachment": "Invoice_copy.docm",
    },
    {
        "from": "DHL Tracking <notice@dhl-hold-release.example>",
        "to": "ruth.abebe@northwind.example",
        "subject": "Package held in Chicago",
        "date": "Fri, 18 Sep 2026 09:41:00 +0000",
        "body": (
            "Hello Ruth,\n\n"
            "A parcel is waiting at the Chicago depot. The label is missing a duty of 48 USD. "
            "The box returns to the sender after 24 hours. "
            "Release the package and confirm your password at "
            "http://dhl-hold-release.example/login\n\n"
            "DHL Tracking"
        ),
    },
]


def make_slots(rng: random.Random, *, phish: bool) -> dict[str, str]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    others = [name for name in FIRST_NAMES if name != first]
    brand = rng.choice(BRANDS)
    if phish and rng.random() < 0.7:
        host = BRAND_HOSTS[brand]
    else:
        host = rng.choice(PHISH_HOSTS if phish else BENIGN_HOSTS)
    domain = host if phish else "northwind.example"
    return {
        "first": first,
        "last": last,
        "full": f"{first} {last}",
        "coworker": rng.choice(others),
        "company": rng.choice(COMPANIES),
        "product": rng.choice(PRODUCTS),
        "dept": rng.choice(DEPTS),
        "team": rng.choice(TEAMS),
        "vendor": rng.choice(VENDORS),
        "bank": rng.choice(BANKS),
        "course": rng.choice(COURSES),
        "brand": brand,
        "amount": rng.choice(AMOUNTS),
        "hours": rng.choice(HOURS),
        "day": rng.choice(DAYS),
        "time": rng.choice(TIMES),
        "city": rng.choice(CITIES),
        "filename": rng.choice(FILENAMES),
        "url": f"http://{host}/{rng.choice(URL_PATHS)}",
        "address": f"{first}.{last}@{domain}".lower(),
        "ticket": f"INC{rng.randint(10000, 99999)}",
    }


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def validate_templates() -> None:
    labels = {family["label"] for family in FAMILIES}
    if labels != {"ham", "spam"}:
        raise ValueError(f"Unexpected labels in templates: {labels}")
    needed: set[str] = set()
    categories = []
    for family in FAMILIES:
        categories.append(family["category"])
        for key in ("subjects", "intros", "details", "asks", "closes"):
            options = family[key]
            if len(options) < 3:
                raise ValueError(f"{family['category']} {key} needs at least 3 templates")
            for template in options:
                needed |= _fields(template)
    if len(categories) != len(set(categories)):
        raise ValueError("Category names must be unique")
    probe = make_slots(random.Random(0), phish=True)
    missing = needed - set(probe)
    if missing:
        raise ValueError(f"Template fields missing from slots: {sorted(missing)}")
    for family in FAMILIES:
        for key in ("subjects", "intros", "details", "asks", "closes"):
            for template in family[key]:
                template.format(**probe)


def render(rng: random.Random, family: dict) -> str:
    slots = make_slots(rng, phish=family["label"] == "spam")
    subject = rng.choice(family["subjects"]).format(**slots)
    details = family["details"]
    primary = rng.choice(details)
    parts = [
        rng.choice(family["intros"]).format(**slots),
        primary.format(**slots),
        rng.choice(family["asks"]).format(**slots),
    ]
    if rng.random() < 0.45 and len(details) > 1:
        extra = rng.choice([item for item in details if item != primary])
        parts.insert(2, extra.format(**slots))
    if rng.random() < 0.85:
        parts.append(rng.choice(family["closes"]).format(**slots))
    body = "\n\n".join(parts)
    return f"Subject: {subject}\n\n{body}"


def build_rows(seed: int = SEED, per_label: int = PER_LABEL) -> list[dict[str, str]]:
    """Return deterministic labeled rows with unique message text."""
    if per_label < 1:
        raise ValueError("per_label must be at least 1")
    validate_templates()
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for label in ("ham", "spam"):
        families = [family for family in FAMILIES if family["label"] == label]
        produced = 0
        guard = 0
        while produced < per_label:
            guard += 1
            if guard > per_label * 100:
                raise RuntimeError(f"Could only build {produced} unique {label} emails")
            family = families[guard % len(families)]
            text = render(rng, family)
            key = " ".join(text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "label": label, "category": family["category"]})
            produced += 1
    rng.shuffle(rows)
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "category"])
        writer.writeheader()
        writer.writerows(rows)


def write_sample_mbox(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    box = mailbox.mbox(path)
    try:
        for item in SAMPLE_MESSAGES:
            message = EmailMessage()
            message["From"] = item["from"]
            message["To"] = item["to"]
            message["Subject"] = item["subject"]
            message["Date"] = item["date"]
            message.set_content(item["body"])
            attachment = item.get("attachment")
            if attachment:
                message.add_attachment(
                    b"synthetic attachment placeholder\n",
                    maintype="application",
                    subtype="octet-stream",
                    filename=attachment,
                )
            box.add(message)
    finally:
        box.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the synthetic spam/ham dataset.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-label", type=int, default=PER_LABEL)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "emails.csv")
    parser.add_argument("--mbox", type=Path, default=ROOT / "data" / "sample_inbox.mbox")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_rows(seed=args.seed, per_label=args.per_label)
    write_csv(rows, args.csv)
    write_sample_mbox(args.mbox)
    labels = Counter(row["label"] for row in rows)
    categories = Counter(row["category"] for row in rows)
    print(f"Wrote {len(rows)} rows to {args.csv}")
    print(f"Labels: {dict(labels)}")
    print(f"Categories: {dict(sorted(categories.items()))}")
    print(f"Wrote {len(SAMPLE_MESSAGES)} sample messages to {args.mbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
