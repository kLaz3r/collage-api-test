import argparse
import asyncio
import os
import random
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import List, Optional, Tuple

import httpx


DEFAULT_URL = "http://localhost:8000/api/collage/create"


@dataclass
class RequestResult:
    ok: bool
    status_code: int
    latency_s: float
    job_id: Optional[str]
    error: Optional[str]


@dataclass
class PollResult:
    ok: bool
    job_id: str
    time_to_complete_s: Optional[float]
    error: Optional[str]


def _randname(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def discover_images(images_dir: Path, max_images: int) -> List[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    candidates: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.tif", "*.tiff"):
        candidates.extend(images_dir.rglob(ext))
    if not candidates:
        raise FileNotFoundError(f"No images found under {images_dir}")
    random.shuffle(candidates)
    return candidates[:max_images]


def build_multipart(files: List[Path]) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    form_parts: List[Tuple[str, Tuple[str, bytes, str]]] = []
    for p in files:
        content_type = (
            "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"]
            else "image/png" if p.suffix.lower() == ".png"
            else "image/tiff" if p.suffix.lower() in [".tif", ".tiff"]
            else "application/octet-stream"
        )
        form_parts.append(("files", (p.name, p.read_bytes(), content_type)))
    return form_parts


async def send_request(client: httpx.AsyncClient, url: str, files: List[Path], extra_form: dict) -> RequestResult:
    form = build_multipart(files)
    for k, v in extra_form.items():
        form.append((k, (None, str(v))))

    start = time.perf_counter()
    try:
        resp = await client.post(url, files=form, timeout=None)
        latency = time.perf_counter() - start
        job_id: Optional[str] = None
        if resp.headers.get("content-type", "").startswith("application/json"):
            try:
                data = resp.json()
                job_id = data.get("job_id")
            except Exception:
                pass
        return RequestResult(ok=resp.is_success, status_code=resp.status_code, latency_s=latency, job_id=job_id, error=None if resp.is_success else resp.text)
    except Exception as e:
        latency = time.perf_counter() - start
        return RequestResult(ok=False, status_code=0, latency_s=latency, job_id=None, error=str(e))


async def poll_until_done(client: httpx.AsyncClient, base_url: str, job_id: str, poll_interval_s: float, timeout_s: float) -> PollResult:
    status_url = base_url.rstrip("/").replace("/create", f"/status/{job_id}")
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            r = await client.get(status_url, timeout=None)
            if not r.is_success:
                await asyncio.sleep(poll_interval_s)
                continue
            js = r.json()
            status = (js.get("status") or "").lower()
            if status in ("completed", "success", "done"):
                created_at = js.get("created_at")
                completed_at = js.get("completed_at")
                # If timestamps available, we can’t easily parse ISO without extra deps; rely on wall time.
                return PollResult(ok=True, job_id=job_id, time_to_complete_s=None, error=None)
            if status in ("failed", "error"):
                return PollResult(ok=False, job_id=job_id, time_to_complete_s=None, error=js.get("error_message") or "failed")
        except Exception:
            pass
        await asyncio.sleep(poll_interval_s)
    return PollResult(ok=False, job_id=job_id, time_to_complete_s=None, error="timeout")


async def worker(name: str, client: httpx.AsyncClient, url: str, images_pool: List[Path], images_per_request: int, extra_form: dict, jobs_out: asyncio.Queue, results_out: asyncio.Queue):
    while True:
        try:
            _ = await jobs_out.get()
        except asyncio.CancelledError:
            break
        files = random.sample(images_pool, k=min(images_per_request, len(images_pool)))
        res = await send_request(client, url, files, extra_form)
        await results_out.put(res)
        jobs_out.task_done()


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def print_summary(latencies: List[float], results: List[RequestResult], wall_s: float):
    total = len(results)
    ok = sum(1 for r in results if r.ok)
    errors = total - ok
    print("=== Benchmark Summary ===")
    print(f"Requests: total={total}, success={ok}, errors={errors}")
    if wall_s > 0:
        print(f"Throughput: {total / wall_s:.2f} req/s")
    if latencies:
        print("Latency (s):")
        print(f"  mean={mean(latencies):.3f}  median={median(latencies):.3f}  p90={percentile(latencies,90):.3f}  p95={percentile(latencies,95):.3f}  p99={percentile(latencies,99):.3f}")


async def run_benchmark(
    url: str,
    images_dir: Path,
    total_requests: int,
    concurrency: int,
    images_per_request: int,
    poll: bool,
    poll_interval_s: float,
    poll_timeout_s: float,
    width_mm: Optional[float],
    height_mm: Optional[float],
    dpi: Optional[int],
    layout_style: Optional[str],
    spacing: Optional[float],
    background_color: Optional[str],
    maintain_aspect_ratio: Optional[bool],
    apply_shadow: Optional[bool],
    output_format: Optional[str],
):
    images_pool = discover_images(images_dir, max_images=1000)
    extra_form = {}
    if width_mm is not None:
        extra_form["width_mm"] = width_mm
    if height_mm is not None:
        extra_form["height_mm"] = height_mm
    if dpi is not None:
        extra_form["dpi"] = dpi
    if layout_style is not None:
        extra_form["layout_style"] = layout_style
    if spacing is not None:
        extra_form["spacing"] = spacing
    if background_color is not None:
        extra_form["background_color"] = background_color
    if maintain_aspect_ratio is not None:
        extra_form["maintain_aspect_ratio"] = str(maintain_aspect_ratio).lower()
    if apply_shadow is not None:
        extra_form["apply_shadow"] = str(apply_shadow).lower()
    if output_format is not None:
        extra_form["output_format"] = output_format

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(None)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        jobs_q: asyncio.Queue = asyncio.Queue()
        results_q: asyncio.Queue = asyncio.Queue()
        for _ in range(total_requests):
            jobs_q.put_nowait(1)

        workers = [asyncio.create_task(worker(f"w{i}", client, url, images_pool, images_per_request, extra_form, jobs_q, results_q)) for i in range(concurrency)]

        results: List[RequestResult] = []
        start_wall = time.perf_counter()
        await jobs_q.join()
        wall_elapsed = time.perf_counter() - start_wall

        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        while not results_q.empty():
            results.append(results_q.get_nowait())

    latencies = [r.latency_s for r in results if r.latency_s is not None]
    print_summary(latencies, results, wall_elapsed)

    if poll:
        job_ids = [r.job_id for r in results if r.ok and r.job_id]
        if not job_ids:
            print("No successful job_ids to poll.")
            return
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            poll_tasks = [poll_until_done(client, url, jid, poll_interval_s, poll_timeout_s) for jid in job_ids]
            poll_results = await asyncio.gather(*poll_tasks)
        succeeded = sum(1 for x in poll_results if x.ok)
        failed = len(poll_results) - succeeded
        print("=== Polling (job completion) ===")
        print(f"Jobs polled: {len(poll_results)}, completed={succeeded}, failed/timeout={failed}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple concurrent benchmark for collage API")
    p.add_argument("--url", default=os.environ.get("BENCH_URL", DEFAULT_URL), help="Create endpoint URL")
    p.add_argument("--images", required=True, type=Path, help="Directory containing sample images")
    p.add_argument("--requests", type=int, default=20, help="Total number of requests")
    p.add_argument("--concurrency", type=int, default=5, help="Concurrent workers")
    p.add_argument("--images-per-request", type=int, default=2, help="Number of images per request (>=2)")
    p.add_argument("--poll", action="store_true", help="Poll job status until completion")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval seconds")
    p.add_argument("--poll-timeout", type=float, default=300.0, help="Polling timeout seconds per job")
    # Optional API form params
    p.add_argument("--width-mm", type=float)
    p.add_argument("--height-mm", type=float)
    p.add_argument("--dpi", type=int)
    p.add_argument("--layout-style", choices=["masonry", "grid"], help="Layout style")
    p.add_argument("--spacing", type=float)
    p.add_argument("--background-color")
    p.add_argument("--maintain-aspect-ratio", action="store_true")
    p.add_argument("--no-maintain-aspect-ratio", dest="maintain_aspect_ratio", action="store_false")
    p.add_argument("--apply-shadow", action="store_true")
    p.add_argument("--no-apply-shadow", dest="apply_shadow", action="store_false")
    p.add_argument("--output-format", choices=["jpeg", "png", "tiff"])
    p.set_defaults(maintain_aspect_ratio=None, apply_shadow=None)
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if args.images_per_request < 2:
        print("--images-per-request must be >= 2", file=sys.stderr)
        return 2
    try:
        asyncio.run(
            run_benchmark(
                url=args.url,
                images_dir=args.images,
                total_requests=args.requests,
                concurrency=args.concurrency,
                images_per_request=args.images_per_request,
                poll=args.poll,
                poll_interval_s=args.poll_interval,
                poll_timeout_s=args.poll_timeout,
                width_mm=args.width_mm,
                height_mm=args.height_mm,
                dpi=args.dpi,
                layout_style=args.layout_style,
                spacing=args.spacing,
                background_color=args.background_color,
                maintain_aspect_ratio=args.maintain_aspect_ratio,
                apply_shadow=args.apply_shadow,
                output_format=args.output_format,
            )
        )
        return 0
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



