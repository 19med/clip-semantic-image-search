# Run the local CLIP search website on Windows

This folder adds a frontend and a Python API to the Part 2 retrieval experiment. The API loads a pretrained CLIP model, searches a saved FAISS image index, and serves the matching local COCO images. The webpage calls `POST /api/search` for any new description typed into the search box.

## 1. Place these files in your existing project folder

Unzip this package into `D:\CLIP_Part_2_Starter`. Keep your downloaded `coco2014` folder and `karpathy_test_image_ids.json` there. The ZIP includes the original `run_clip.py`, so overwrite it only if you have not edited your local copy. Your existing test results can stay in `results`.

The resulting layout should include:

```text
D:\CLIP_Part_2_Starter\
  app.py
  build_index.py
  run_clip.py
  karpathy_test_image_ids.json
  coco2014\images\val2014\COCO_val2014_000000000042.jpg
  web\index.html
```

Make sure all five previously missing images are now present: `58636`, `224742`, `252101`, `459590`, and `562582`.

## 2. Install the web server packages

In PowerShell:

```powershell
cd D:\CLIP_Part_2_Starter
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn
```

Your existing environment already has PyTorch, CLIP, NumPy, Pillow, and `faiss-cpu` from the retrieval experiment.

## 3. Build searchable indexes once

The earlier experiment stored only rankings, not reusable image indexes. These two commands encode all 5,000 test images and save an index for each CLIP backbone:

```powershell
.\.venv\Scripts\python.exe .\build_index.py --image-root .\coco2014 --ids .\karpathy_test_image_ids.json --model ViT-B/16 --index-dir .\indexes
.\.venv\Scripts\python.exe .\build_index.py --image-root .\coco2014 --ids .\karpathy_test_image_ids.json --model ViT-B/32 --index-dir .\indexes
```

Do **not** run `run_clip.py` again for the website. It is for the already-completed evaluation. The saved indexes preserve the exact order of `karpathy_test_image_ids.json` and contain normalized image vectors.

## 4. Start the API and webpage

```powershell
.\.venv\Scripts\python.exe .\app.py --image-root .\coco2014 --ids .\karpathy_test_image_ids.json --index-dir .\indexes
```

Keep PowerShell open. Visit **http://127.0.0.1:8000** in your browser. Type a description, choose ViT-B/16 or ViT-B/32, and click Search. To stop the server, press **Ctrl+C**. API documentation is available at **http://127.0.0.1:8000/docs**.

## API contract for your teammate

```http
POST /api/search
Content-Type: application/json

{"query":"a dog sleeping on shoes","model":"ViT-B/16","top_k":10}
```

Response fields: `query`, `model`, `gallery_images`, and `results`. Each result contains `image_id`, `score`, and `image_url`. GET `/api/health` reports the available indexes. GET `/api/images/{image_id}` serves only images in the 5,000-image gallery.

The current interface supports **live CLIP search** for both backbones. The TF-IDF caption baseline is maintained by Part 1 and is **not** offered as a live search option here because its indexing code and captions were not provided. To offer it in the same website, Part 1 can expose its own text search using the same result fields and add it as another model choice.

This is a local demonstration, bound to `127.0.0.1`. Deployment to a public website needs hosting for the model weights, image files, and API; `127.0.0.1` is not accessible from another computer.
