# garden-app

### Requirement: Health check

The service SHALL expose `GET /health` and return HTTP 200 when the process can serve requests. The body SHALL say whether mock mode is on and whether the model is ready. The body SHALL NOT include `TINKER_API_KEY`.

#### Scenario: Process is up in mock mode

- **WHEN** the service is running with `FIELDSIGHT_MOCK=1`
- **THEN** `GET /health` returns HTTP 200
- **AND** the body reports mock mode and a ready model

#### Scenario: Model is still loading

- **WHEN** mock mode is off and the sampling client has not finished loading
- **THEN** `GET /health` still returns HTTP 200
- **AND** the body reports that the model is not ready

### Requirement: Diagnose one leaf photo

The service SHALL accept one image on `POST /api/diagnose` and return JSON with the plant, the condition, three care steps when the label is known, sources, and the disclaimer from `care/fixes.json`. Confidence SHALL be present only as a pass-through of a numeric value the model returned, and SHALL be null when the model does not supply one. Mock mode SHALL return that shape without calling Tinker. A photo that is empty, unreadable, or over the size limit SHALL be rejected before any model call.

#### Scenario: Readable photo in mock mode

- **WHEN** a client posts a readable image and mock mode is on
- **THEN** the response is HTTP 200
- **AND** it includes a known label, a plant, a condition, exactly three steps, sources, and the disclaimer
- **AND** confidence is null
- **AND** the service has not imported the Tinker SDK

#### Scenario: Same photo is stable in mock mode

- **WHEN** the same image bytes are posted twice in mock mode
- **THEN** both responses name the same label

#### Scenario: Unreadable upload

- **WHEN** a client posts bytes that are not an image
- **THEN** the response is HTTP 400
- **AND** no diagnosis is counted

#### Scenario: Oversized upload

- **WHEN** a client posts an image larger than the configured byte limit
- **THEN** the response is HTTP 413

#### Scenario: Model is not ready

- **WHEN** mock mode is off and the sampling client is not loaded
- **THEN** a readable photo receives HTTP 503

### Requirement: Rate limit

The service SHALL limit diagnosis posts per client IP and across the process. When a limit is exceeded it SHALL return HTTP 429 and SHALL NOT count another diagnosis.

#### Scenario: Too many photos from one client

- **WHEN** a client exceeds the configured per-window diagnosis limit
- **THEN** the next post returns HTTP 429

### Requirement: Anonymous daily counts

The service SHALL record the label, the latency, and the timestamp of a successful diagnosis, and SHALL NOT record image bytes. When `DATABASE_URL` is set and Postgres is reachable, those rows SHALL be stored in Postgres. When it is unset or unreachable, the service SHALL keep the count in memory and diagnosis SHALL still succeed. `GET /api/stats` SHALL return how many diagnoses have been recorded for the current UTC day.

#### Scenario: Count increases after a diagnosis

- **WHEN** a readable photo is diagnosed
- **THEN** `GET /api/stats` reports one more diagnosis for today than it did before
- **AND** the stored record has a label, a latency, and a timestamp
- **AND** the stored record has no image field

#### Scenario: No database configured

- **WHEN** `DATABASE_URL` is unset
- **THEN** diagnosis and `GET /api/stats` succeed using in-memory counts

#### Scenario: Database cannot be reached

- **WHEN** `DATABASE_URL` is set and the connection fails
- **THEN** the service starts with in-memory counts
- **AND** diagnosis still succeeds

### Requirement: Garden page

The service SHALL serve a mobile-first page with a rear-camera capture control (`capture="environment"`), a result card for the plant, condition, and three steps, a control that reads the result aloud with the browser Speech Synthesis API, and a finish state reached by "Done, go outside". The page SHALL include the results chart, the accuracy and cost table, PlantDoc Creative Commons Attribution 4.0 credit, and the extension-service disclaimer.

#### Scenario: Page contains the garden flow

- **WHEN** a browser requests `/`
- **THEN** the HTML includes rear-camera capture, a read-aloud control using speech synthesis, and the text "Done, go outside"
- **AND** it includes the results chart, the fine-tuned accuracy, PlantDoc attribution, and extension-source guidance

### Requirement: Render blueprint

The repository SHALL contain a `render.yaml` blueprint for a Python web service on the starter plan with health check path `/health`, `TINKER_API_KEY` set to `sync: false`, and `FIELDSIGHT_MODEL_PATH`. It SHALL offer an optional Postgres database. The blueprint SHALL validate against the Render blueprint schema.

#### Scenario: Blueprint matches the deploy contract

- **WHEN** `render.yaml` is parsed
- **THEN** it defines a Python starter web service with `healthCheckPath: /health`
- **AND** `TINKER_API_KEY` has `sync: false` and no committed value
- **AND** `FIELDSIGHT_MODEL_PATH` is set
- **AND** a Postgres database is defined
- **AND** the document validates against the Render blueprint schema
