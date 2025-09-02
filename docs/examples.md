# Usage Examples

Practical examples for integrating with the Collage Maker API using different programming languages and tools.

## Python Examples

### Basic Collage Creation

```python
import requests

def create_basic_collage(image_paths, output_path):
    """Create a basic collage with default settings"""

    # Prepare the multipart form data
    files = []
    for i, image_path in enumerate(image_paths):
        files.append(('files', (f'image_{i}.jpg', open(image_path, 'rb'), 'image/jpeg')))

    # API endpoint
    url = 'http://localhost:8000/api/collage/create'

    # Make the request
    response = requests.post(url, files=files)

    if response.status_code == 200:
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"Collage job started: {job_id}")

        # Poll for completion
        return poll_job_status(job_id, output_path)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False

def poll_job_status(job_id, output_path, max_attempts=30):
    """Poll job status until completion"""

    import time

    for attempt in range(max_attempts):
        response = requests.get(f'http://localhost:8000/api/collage/status/{job_id}')

        if response.status_code == 200:
            status_data = response.json()

            if status_data['status'] == 'completed':
                # Download the collage
                download_response = requests.get(f'http://localhost:8000/api/collage/download/{job_id}')

                if download_response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(download_response.content)
                    print(f"Collage saved to: {output_path}")
                    return True
                else:
                    print("Failed to download collage")
                    return False

            elif status_data['status'] == 'failed':
                print(f"Job failed: {status_data.get('error_message', 'Unknown error')}")
                return False

            else:
                print(f"Job status: {status_data['status']} ({status_data['progress']}%)")
                time.sleep(2)  # Wait 2 seconds before next check

        else:
            print(f"Failed to check status: {response.status_code}")
            return False

    print("Timeout waiting for job completion")
    return False

# Usage example
if __name__ == "__main__":
    images = ["photo1.jpg", "photo2.jpg", "photo3.jpg", "photo4.jpg"]
    create_basic_collage(images, "my_collage.jpg")
```

### Advanced Configuration Example

```python
import requests
from typing import List, Optional

class CollageClient:
    """Advanced client for the Collage Maker API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def create_custom_collage(
        self,
        image_paths: List[str],
        width_inches: float = 12,
        height_inches: float = 18,
        dpi: int = 150,
        layout_style: str = "masonry",
        spacing: int = 10,
        background_color: str = "#FFFFFF",
        maintain_aspect_ratio: bool = True,
        apply_shadow: bool = False
    ) -> Optional[str]:
        """Create a collage with custom configuration"""

        # Validate inputs
        if len(image_paths) < 2:
            raise ValueError("At least 2 images required")
        if len(image_paths) > 200:
            raise ValueError("Maximum 200 images allowed")

        # Prepare files
        files = []
        for i, image_path in enumerate(image_paths):
            try:
                with open(image_path, 'rb') as f:
                    files.append(('files', (f'image_{i}.jpg', f, 'image/jpeg')))
            except FileNotFoundError:
                print(f"Warning: Image not found: {image_path}")
                continue

        if len(files) < 2:
            raise ValueError("At least 2 valid images required")

        # Prepare data
        data = {
            'width_inches': width_inches,
            'height_inches': height_inches,
            'dpi': dpi,
            'layout_style': layout_style,
            'spacing': spacing,
            'background_color': background_color,
            'maintain_aspect_ratio': maintain_aspect_ratio,
            'apply_shadow': apply_shadow
        }

        # Make request
        url = f"{self.base_url}/api/collage/create"
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            return response.json()['job_id']
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def get_job_status(self, job_id: str) -> dict:
        """Get job status"""
        url = f"{self.base_url}/api/collage/status/{job_id}"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Status check failed: {response.status_code}")

    def download_collage(self, job_id: str, output_path: str) -> bool:
        """Download completed collage"""
        url = f"{self.base_url}/api/collage/download/{job_id}"
        response = requests.get(url)

        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False

    def wait_for_completion(self, job_id: str, timeout: int = 300) -> dict:
        """Wait for job completion with timeout"""
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            if status['status'] in ['completed', 'failed']:
                return status

            time.sleep(2)

        raise TimeoutError("Job completion timeout")

# Usage examples
client = CollageClient()

# High-quality print collage
job_id = client.create_custom_collage(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    width_inches=16,
    height_inches=20,
    dpi=300,
    layout_style="masonry",
    spacing=15,
    background_color="#FFFFFF"
)

# Wait for completion and download
status = client.wait_for_completion(job_id)
if status['status'] == 'completed':
    client.download_collage(job_id, "print_collage.jpg")
```

## JavaScript/Node.js Examples

### Basic Usage with Axios

```javascript
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

async function createCollage(imagePaths, outputPath) {
    try {
        // Create form data
        const formData = new FormData();

        // Add images
        for (let i = 0; i < imagePaths.length; i++) {
            const imageStream = fs.createReadStream(imagePaths[i]);
            formData.append("files", imageStream, `image_${i}.jpg`);
        }

        // Make request
        const response = await axios.post(
            "http://localhost:8000/api/collage/create",
            formData,
            {
                headers: formData.getHeaders(),
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
            }
        );

        const jobId = response.data.job_id;
        console.log(`Job started: ${jobId}`);

        // Wait for completion
        await waitForJob(jobId);

        // Download result
        const downloadResponse = await axios.get(
            `http://localhost:8000/api/collage/download/${jobId}`,
            {
                responseType: "stream",
            }
        );

        const writer = fs.createWriteStream(outputPath);
        downloadResponse.data.pipe(writer);

        return new Promise((resolve, reject) => {
            writer.on("finish", resolve);
            writer.on("error", reject);
        });
    } catch (error) {
        console.error(
            "Error creating collage:",
            error.response?.data || error.message
        );
        throw error;
    }
}

async function waitForJob(jobId, maxAttempts = 30) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
            const response = await axios.get(
                `http://localhost:8000/api/collage/status/${jobId}`
            );
            const status = response.data;

            if (status.status === "completed") {
                return status;
            } else if (status.status === "failed") {
                throw new Error(`Job failed: ${status.error_message}`);
            }

            console.log(`Status: ${status.status} (${status.progress}%)`);
            await new Promise((resolve) => setTimeout(resolve, 2000));
        } catch (error) {
            console.error(`Status check failed: ${error.message}`);
            throw error;
        }
    }
    throw new Error("Timeout waiting for job completion");
}

// Usage
createCollage(["photo1.jpg", "photo2.jpg", "photo3.jpg"], "output.jpg")
    .then(() => console.log("Collage created successfully"))
    .catch((error) =>
        console.error("Failed to create collage:", error.message)
    );
```

### React Web Application Example

```jsx
import React, { useState, useRef } from "react";
import axios from "axios";

function CollageCreator() {
    const [images, setImages] = useState([]);
    const [jobId, setJobId] = useState(null);
    const [status, setStatus] = useState(null);
    const [loading, setLoading] = useState(false);
    const fileInputRef = useRef(null);

    const handleFileSelect = (event) => {
        const files = Array.from(event.target.files);
        setImages(files);
    };

    const createCollage = async () => {
        if (images.length < 2) {
            alert("Please select at least 2 images");
            return;
        }

        setLoading(true);

        try {
            const formData = new FormData();

            // Add images
            images.forEach((file, index) => {
                formData.append("files", file);
            });

            // Add configuration
            formData.append("width_inches", "12");
            formData.append("height_inches", "18");
            formData.append("layout_style", "masonry");

            const response = await axios.post(
                "http://localhost:8000/api/collage/create",
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }
            );

            setJobId(response.data.job_id);
            pollStatus(response.data.job_id);
        } catch (error) {
            console.error("Error creating collage:", error);
            alert("Failed to create collage");
        } finally {
            setLoading(false);
        }
    };

    const pollStatus = async (jobId) => {
        try {
            const response = await axios.get(
                `http://localhost:8000/api/collage/status/${jobId}`
            );
            setStatus(response.data);

            if (response.data.status === "completed") {
                // Download the collage
                const downloadResponse = await axios.get(
                    `http://localhost:8000/api/collage/download/${jobId}`,
                    {
                        responseType: "blob",
                    }
                );

                const url = window.URL.createObjectURL(
                    new Blob([downloadResponse.data])
                );
                const link = document.createElement("a");
                link.href = url;
                link.setAttribute("download", "collage.jpg");
                document.body.appendChild(link);
                link.click();
                link.remove();
            } else if (response.data.status === "failed") {
                alert(`Job failed: ${response.data.error_message}`);
            } else {
                // Continue polling
                setTimeout(() => pollStatus(jobId), 2000);
            }
        } catch (error) {
            console.error("Error checking status:", error);
        }
    };

    return (
        <div className="collage-creator">
            <h2>Create Photo Collage</h2>

            <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: "none" }}
            />

            <button onClick={() => fileInputRef.current.click()}>
                Select Images ({images.length} selected)
            </button>

            <button
                onClick={createCollage}
                disabled={loading || images.length < 2}
            >
                {loading ? "Creating..." : "Create Collage"}
            </button>

            {status && (
                <div className="status">
                    <p>Status: {status.status}</p>
                    <p>Progress: {status.progress}%</p>
                </div>
            )}
        </div>
    );
}

export default CollageCreator;
```

## cURL Examples

### Basic Collage Creation

```bash
# Create a simple collage with default settings
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "files=@photo3.jpg" \
  -F "files=@photo4.jpg"
```

### Custom Configuration

```bash
# Create a high-quality print collage
curl -X POST "http://localhost:8000/api/collage/create" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "width_inches=16" \
  -F "height_inches=20" \
  -F "dpi=300" \
  -F "layout_style=masonry" \
  -F "spacing=15" \
  -F "background_color=#FFFFFF" \
  -F "maintain_aspect_ratio=true" \
  -F "apply_shadow=false"
```

### Check Job Status

```bash
# Check the status of a job
curl "http://localhost:8000/api/collage/status/123e4567-e89b-12d3-a456-426614174000"
```

### Download Completed Collage

```bash
# Download the finished collage
curl -o "my_collage.jpg" \
  "http://localhost:8000/api/collage/download/123e4567-e89b-12d3-a456-426614174000"
```

### List All Jobs

```bash
# Get a list of all jobs
curl "http://localhost:8000/api/collage/jobs"
```

### Cleanup Job Files

```bash
# Clean up temporary files for a job
curl -X DELETE \
  "http://localhost:8000/api/collage/cleanup/123e4567-e89b-12d3-a456-426614174000"
```

## PHP Example

```php
<?php

class CollageAPI {
    private $baseUrl;

    public function __construct($baseUrl = 'http://localhost:8000') {
        $this->baseUrl = rtrim($baseUrl, '/');
    }

    public function createCollage($imagePaths, $config = []) {
        if (count($imagePaths) < 2) {
            throw new Exception('At least 2 images required');
        }

        // Prepare multipart data
        $boundary = '----FormBoundary' . md5(time());
        $data = '';

        // Add images
        foreach ($imagePaths as $index => $path) {
            if (!file_exists($path)) {
                continue;
            }

            $filename = basename($path);
            $fileContents = file_get_contents($path);

            $data .= "--$boundary\r\n";
            $data .= "Content-Disposition: form-data; name=\"files\"; filename=\"$filename\"\r\n";
            $data .= "Content-Type: image/jpeg\r\n\r\n";
            $data .= $fileContents . "\r\n";
        }

        // Add configuration parameters
        $defaultConfig = [
            'width_inches' => 12,
            'height_inches' => 18,
            'dpi' => 150,
            'layout_style' => 'masonry',
            'spacing' => 10,
            'background_color' => '#FFFFFF',
            'maintain_aspect_ratio' => true,
            'apply_shadow' => false
        ];

        $config = array_merge($defaultConfig, $config);

        foreach ($config as $key => $value) {
            $data .= "--$boundary\r\n";
            $data .= "Content-Disposition: form-data; name=\"$key\"\r\n\r\n";
            $data .= $value . "\r\n";
        }

        $data .= "--$boundary--\r\n";

        // Make HTTP request
        $context = stream_context_create([
            'http' => [
                'method' => 'POST',
                'header' => "Content-Type: multipart/form-data; boundary=$boundary",
                'content' => $data
            ]
        ]);

        $response = file_get_contents($this->baseUrl . '/api/collage/create', false, $context);

        if ($response === false) {
            throw new Exception('Failed to create collage');
        }

        $result = json_decode($response, true);

        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new Exception('Invalid JSON response');
        }

        return $result;
    }

    public function getJobStatus($jobId) {
        $response = file_get_contents($this->baseUrl . "/api/collage/status/$jobId");

        if ($response === false) {
            throw new Exception('Failed to get job status');
        }

        return json_decode($response, true);
    }

    public function downloadCollage($jobId, $outputPath) {
        $response = file_get_contents($this->baseUrl . "/api/collage/download/$jobId");

        if ($response === false) {
            throw new Exception('Failed to download collage');
        }

        return file_put_contents($outputPath, $response) !== false;
    }

    public function waitForCompletion($jobId, $timeout = 300) {
        $startTime = time();

        while (time() - $startTime < $timeout) {
            $status = $this->getJobStatus($jobId);

            if ($status['status'] === 'completed') {
                return $status;
            } elseif ($status['status'] === 'failed') {
                throw new Exception('Job failed: ' . ($status['error_message'] ?? 'Unknown error'));
            }

            sleep(2);
        }

        throw new Exception('Timeout waiting for job completion');
    }
}

// Usage example
try {
    $api = new CollageAPI();

    // Create collage
    $result = $api->createCollage([
        'photo1.jpg',
        'photo2.jpg',
        'photo3.jpg'
    ], [
        'width_inches' => 16,
        'height_inches' => 20,
        'layout_style' => 'grid'
    ]);

    $jobId = $result['job_id'];
    echo "Job started: $jobId\n";

    // Wait for completion
    $status = $api->waitForCompletion($jobId);
    echo "Job completed!\n";

    // Download result
    $api->downloadCollage($jobId, 'collage.jpg');
    echo "Collage saved as collage.jpg\n";

} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
}

?>
```

## Batch Processing Example

```python
import requests
import os
from pathlib import Path
import time

def process_batch_collages(input_dir, output_dir, batch_size=10):
    """Process multiple collages from a directory of images"""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    all_images = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if len(all_images) < 2:
        print("Not enough images found")
        return

    # Process in batches
    jobs = []
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i + batch_size]

        if len(batch) < 2:
            continue

        print(f"Processing batch {len(jobs) + 1}: {len(batch)} images")

        # Create collage for this batch
        job_id = create_batch_collage(batch, output_path / f"collage_batch_{len(jobs) + 1}.jpg")
        if job_id:
            jobs.append(job_id)

    # Wait for all jobs to complete
    completed_jobs = []
    for job_id in jobs:
        try:
            status = wait_for_job_completion(job_id)
            if status['status'] == 'completed':
                completed_jobs.append(job_id)
                print(f"Batch job {job_id} completed")
            else:
                print(f"Batch job {job_id} failed: {status.get('error_message')}")
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")

    print(f"Successfully completed {len(completed_jobs)} out of {len(jobs)} batches")

def create_batch_collage(image_paths, output_path):
    """Create a collage from a batch of images"""

    files = []
    for image_path in image_paths:
        try:
            files.append(('files', (image_path.name, open(image_path, 'rb'), 'image/jpeg')))
        except Exception as e:
            print(f"Warning: Could not open {image_path}: {e}")
            continue

    if len(files) < 2:
        print("Not enough valid images in batch")
        return None

    try:
        response = requests.post(
            'http://localhost:8000/api/collage/create',
            files=files,
            data={
                'layout_style': 'masonry',
                'width_inches': 12,
                'height_inches': 18,
                'dpi': 150
            }
        )

        if response.status_code == 200:
            job_id = response.json()['job_id']

            # Download immediately for batch processing
            download_response = requests.get(f'http://localhost:8000/api/collage/download/{job_id}')

            if download_response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(download_response.content)
                print(f"Saved: {output_path}")
                return job_id
            else:
                print(f"Failed to download collage for job {job_id}")
                return None
        else:
            print(f"Failed to create collage: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error creating collage: {e}")
        return None
    finally:
        # Close all file handles
        for _, file_tuple in files:
            file_tuple[1].close()

def wait_for_job_completion(job_id, timeout=300):
    """Wait for a job to complete"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f'http://localhost:8000/api/collage/status/{job_id}')
            status = response.json()

            if status['status'] in ['completed', 'failed']:
                return status

            time.sleep(1)
        except Exception as e:
            print(f"Error checking status for job {job_id}: {e}")
            break

    return {'status': 'timeout'}

# Usage
if __name__ == "__main__":
    process_batch_collages(
        input_dir="input_images",
        output_dir="output_collagess",
        batch_size=15
    )
```
