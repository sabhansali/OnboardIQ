# OnboardIQ

**Intelligent SaaS Onboarding and Implementation Assistance System**

OnboardIQ is a Streamlit application that processes synthetic customer-onboarding requests, classifies intent/complexity/priority, retrieves approved onboarding guidance, routes each case to controlled automated handling or human review, and tracks tickets, notifications, retries, SLA deadlines, and audit events in SQLite.

It adapts the teacher's Intelligent Customer Support with Escalation workflow to the SaaS onboarding and implementation subdomain. It is a college/lab demonstration, not a production customer-support platform.

## Included capabilities

- CSV, JSON, and simulated-chat request intake.
- Input validation, whitespace normalization, malformed-input handling, duplicate-ID detection, and persisted rejected-row reasons.
- Rule-based classification with optional Groq-assisted structured classification. High-risk rules override the LLM result.
- Hybrid keyword + TF-IDF/cosine knowledge retrieval against **approved** articles only. Match score is a heuristic, not a calibrated probability.
- Risk-first routing to auto-resolution, product feedback, human review, specialist escalation, or urgent incident response.
- SQLite persistence for users, import batches, raw requests, tickets, ticket events, audit events, notifications, and individual notification attempts.
- Human-in-the-loop review actions: approve, edit, hold, request information, escalate, resolve, reopen, assign, and start work.
- Password-hashed local demo accounts, role-based authorization, ticket-version checks, allowed state transitions, and audit records for denied/stale actions.
- Simulated email/in-app notifications; individual retry attempts; exponential backoff; idempotent sent state; failed-notification queue and dead-letter state.
- First-response and resolution deadlines, SLA-at-risk/breach states, authorized hold/resume clock accounting, and support-lead alerts.
- Analytics including classification accuracy, per-class precision/recall/F1, macro-F1, knowledge-base resolution, escalation, notification success, SLA compliance, processing time, review completion, retry recovery, and audit completeness.
- Automated tests for normal, boundary, invalid, ambiguous, security-sensitive, and failure cases.

## Windows setup (PowerShell)

Use Python 3.10 or newer (Python 3.11/3.12 is a good choice for a student project). In VS Code, open the `OnboardIQ` folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, you can invoke the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Start the application:

```powershell
streamlit run app.py
```

Streamlit displays a local URL in the terminal (normally `http://localhost:8501`). Keep that terminal open while using the app.

## Demo sign-in

The first local database initialization creates these demo users:

| Role | Username | Password |
|---|---|---|
| Administrator | `admin` | `AdminDemo!2026` |
| Manager | `manager` | `ManagerDemo!2026` |
| Specialist | `specialist` | `SpecialistDemo!2026` |

These are public demo credentials for local assessment only. To change them, edit the corresponding values in `.env` **before the first run / before the users are seeded**. The app uses salted PBKDF2 password hashes in SQLite. This is a lab-grade local login model, not a substitute for an identity provider, HTTPS, account lockout, or a production security review.

## Optional Groq classifier

**Groq is optional.** The default application and full test suite run with deterministic rule-based classification and local semantic retrieval. No API key is required for the baseline project.

To enable the optional Groq variant:

1. Obtain your own Groq API key.
2. Put it in the local `.env` file (never commit or share the key):

   ```dotenv
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   USE_GROQ_CLASSIFIER=true
   ```

3. Restart Streamlit and enable the Groq option in the sidebar when available.

The model responds in structured JSON; output values are validated against the permitted labels. A call failure falls back to the rule-based classifier. Local critical-risk rules still take priority over the model output. API model availability can change, so a different model can be supplied through `GROQ_MODEL` without changing source code.

## Workflow walkthrough

1. **Overview** — review ticket counts, priority mix, SLA status, and the cases requiring attention.
2. **Intake & Processing** — load `data/sample_requests.csv` or `data/sample_requests.json`, upload your own CSV/JSON file, or submit a simulated chat request. Invalid entries are rejected with recorded reasons rather than processed as valid tickets.
3. **Ticket Queue** — filter by status/priority, inspect classification and matched article, edit response drafts, and apply role-authorized actions. The app re-checks the user's active status/role and ticket version at the backend boundary.
4. **Notifications** — inspect draft/sent/failed events and delivery-attempt records. An administrator/manager can run the controlled delivery failure lab to demonstrate first-attempt success, fail-twice-then-succeed, total failure, retry, and dead-letter handling. Delivery is simulated and sends no real email or chat message.
5. **Knowledge Base** — inspect approved onboarding guidance and test retrieval against the local article set.
6. **Analytics & Evaluation** — inspect persisted workflow metrics and per-class classification reports; compare keyword-only retrieval with hybrid semantic retrieval; optionally compare rules and Groq classification if configured; run tests.
7. **Audit & Security** — inspect events and ticket history, and download CSV exports.
8. **User Management** — administrator-only creation/deactivation of local demo users.

## Input contract

Every CSV/JSON request needs these fields:

- `request_id` — unique request identifier within the upload and across persisted requests.
- `customer_id` — synthetic customer identifier (do not put sensitive customer data in lab files).
- `company_name` — company/tenant name (use synthetic values).
- `contact_email` — syntactically valid address (use reserved `.test` addresses in samples).
- `request_text` — non-empty onboarding or implementation request.
- `created_at` — ISO-8601 timestamp.

JSON input can be either a list of objects or an object containing a `requests` list. The sample dataset intentionally includes malformed rows: an empty query, missing contact email, and duplicate request ID. They are expected to appear in the rejected-record view rather than enter ticket processing. Valid sample rows contain `expected_intent`, `expected_complexity`, and `expected_priority` fields for evaluation; those labels are ground truth for evaluation and are not passed to the classifier.

## Routing safeguards

- Critical security concerns or widespread access/service impact are escalated before the routine-match logic.
- Only an approved knowledge article can contribute a response draft.
- A generated proposal is not counted as a completed resolution by itself.
- Higher-risk and non-eligible routes require human approval before the customer-facing response is dispatched.
- Notification templates are simulated. A ticket receipt or assignment acknowledgement is distinct from an approval-gated customer-resolution response.
- Status transitions and actions are validated in the backend and persisted transactionally. Stale UI forms are rejected instead of overwriting a newer ticket update.
- `P1`–`P4` SLA targets in `config.json` are **illustrative lab values**, not contractual recommendations.

## Run tests

From the project root, activate the virtual environment and run:

```powershell
python -m pytest -q tests
```

Or run the root wrapper:

```powershell
python test_support.py
```

The tests use temporary SQLite databases and do not alter the normal `onboardiq.db`. They do not require a Groq API key, external network, or actual messaging credentials.

## Database and generated files

- `onboardiq.db` is created locally on first app start and is excluded from the project archive/source control.
- `.env` contains local secrets/configuration and is excluded from source control.
- The app creates CSV exports under `exports/` only when you request them.
- To reset a disposable lab instance, stop Streamlit and delete `onboardiq.db`. The next startup recreates the schema and seeds demo users. This also deletes all local tickets and audit history, so export anything you need first.

## Project structure

```text
OnboardIQ/
├── app.py
├── config.json
├── knowledge_base.json
├── requirements.txt
├── .env.example
├── README.md
├── core/
│   ├── classifier.py
│   ├── config.py
│   ├── database.py
│   ├── ingestion.py
│   ├── metrics.py
│   ├── notifications.py
│   ├── resolver.py
│   ├── security.py
│   ├── sla_monitor.py
│   ├── sla_service.py
│   ├── workflow.py
│   └── workflow_router.py
├── data/
│   ├── sample_requests.csv
│   └── sample_requests.json
├── tests/
│   └── test_support.py
└── test_support.py
```

## Known scope limits

- Notifications simulate email/in-app delivery and are not connected to Gmail, Slack, Microsoft Teams, or an SMTP provider.
- Semantic retrieval uses local TF-IDF/cosine similarity rather than a hosted embedding API, so no vector database or embedding key is required.
- SLA calculations use elapsed time and illustrative minute targets; business calendars, holidays, and enterprise on-call scheduling are outside this lab version.
- The local account model is designed for demonstration and must not be exposed publicly without a proper production security architecture.
