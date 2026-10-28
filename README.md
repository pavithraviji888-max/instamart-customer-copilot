# Instamart Customer Intelligence & Support Copilot

A reproducible, offline capstone implementation using the supplied synthetic data generator and `mock_llm.py`.

## Before Running

1. Copy the **unmodified** `generate_data.py` and `mock_llm.py` from your capstone brief into the repository root.
2. Use Python 3.11 or later.
3. The project makes no network calls and requires no API key.

## Overview

* **Part A:** Cleans synthetic customer and order data, creates clean CSV files and SQLite tables, runs seven SQL analyses, and validates key checkpoints.
* **Part B:** Creates RFM customer segments and trains lapse prediction models.
* **Part C:** Redacts support tickets, classifies tickets using the offline mock model, retries transient errors, validates outputs, applies token budgets, and routes tickets.
* **Part D:** Chunks and retrieves policy text, evaluates grounded answers, and runs a tool-using refund agent with limits and checkpoints.

## Reproduce

Run these commands in PowerShell from the repository root. If your virtual environment is already active, you can skip the first two commands.

1. Create a virtual environment:

   `python -m venv .venv`

2. Activate it:

   `.\.venv\Scripts\Activate.ps1`

3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Generate the synthetic data:

   `python generate_data.py`

5. Run the complete pipeline:

   `python -m src.run_all`

The scripts write generated files under `data/`, `outputs/`, `state/`, and `sql/`. Run commands from the repository root so relative paths resolve correctly.

## Part A

### Cleaning Rules

* Customer city values are stripped and title-cased.
* Gender variants are mapped to `male`, `female`, or `unknown`.
* Ages outside 10–100 are treated as missing and then filled using the city median.
* Signup dates are parsed and saved as ISO-formatted text.
* Order timestamps are parsed, exact duplicates are removed, and the latest row per order ID is retained.
* Invalid order amounts are treated as missing and recalculated using product price, quantity, and discount.
* Negative delivery times are treated as missing.
* Missing delivery times for completed orders are filled using the median for the relevant product category.
* Cancelled-order delivery times remain missing.
* Cleaned customers, products, and orders are saved as CSV files and SQLite tables.

The validation script checks core reference counts and prints `ALL CHECKPOINTS PASSED` when its assertions pass.

The seven analytical SQL queries are stored in `sql/queries.sql`.

## Part B

RFM analysis uses completed orders and the snapshot date `2025-10-01`.

K-means clustering is evaluated for K values from 2 through 8 using scaled RFM features and a fixed random seed.

The lapse model uses features from completed and final orders before `2025-09-01`. The September window is used only to create lapse labels.

Preprocessing is fitted within the training pipeline, and the test set is held out for evaluation. Segment labels are not used as prediction features.

The Premium retention list is illustrative. Its Premium flag is derived from whole-window RFM segmentation, so it is not a leakage-free evaluation of premium status at the prediction cutoff.

## Part C

* Tickets are processed in source-file order.
* Empty tickets and tickets exceeding 400 tokens are skipped or flagged before model calls.
* Email addresses, phone numbers, and order IDs are redacted before prompts are sent to the mock model.
* Responses are validated as JSON or through keyword-based validation.
* Invalid responses are flagged for human review.
* Retries use exponential backoff with jitter.
* Fraud-related tickets are routed to security.
* Premium customers are routed to the premium desk.
* Other valid tickets are routed to support.

## Part D

Policy documents are divided into 40-word chunks with 10-word overlap.

TF-IDF and cosine similarity are used to retrieve relevant policy chunks. The assistant generates answers using retrieved context.

The refund agent uses database and policy tools, enforces iteration, token, and time limits, writes a JSONL refund ledger, and saves checkpoints after tool observations.

The supplied `mock_llm.py` is the model interface. Refund actions are simulated and do not move real money.

## Limitations and Next Steps

* All records and policies are synthetic and fictional. They do not represent actual Swiggy Instamart data or official company policies.
* TF-IDF retrieval is lexical and may miss semantically equivalent wording.
* The supplied offline mock model is deterministic and is not a production LLM.
* Refund actions are simulated ledger entries, not real payment transactions.
* The retention list is illustrative because its Premium flag comes from whole-window RFM and therefore uses post-cutoff information, even though the lapse-model features use only pre-cutoff data.
* Production deployment would require human review, monitoring, access controls, comprehensive testing, real policy governance, and secure payment integrations.

