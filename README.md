# RTSPVision

Watches an RTSP camera feed, detects people with YOLOv8, and logs each time
someone enters a defined zone—saving a timestamp and a cropped image of
the person.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
RTSP_URL=rtsp://user:pass@192.168.x.x:554/Streaming/Channels/102
```

## Run

```bash
python main.py
```

Press `q` to quit.

## Output

Each zone entry is saved to `captures/`:

- `entry_XXXX_<timestamp>.jpg` – Cropped snapshot of the person.
- `entries_log.csv` – `entry_id, timestamp, image_path`.
