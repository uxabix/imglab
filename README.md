# imglab — Image Processing Educational Project

This repository contains an **image-processing project** that started as a university assignment and was further developed into a more complete, extensible, and practical tool.  
The goal is to **implement common image-processing operations manually**, using only core Python and **NumPy**, without relying on external image-processing libraries like OpenCV.

The project demonstrates how classical image transformations work internally: convolution, filtering, edge detection, pixel-wise operations, gamma correction, and more.

---

## ✨ Features

### **1. Arithmetic Operations**
- Add  
- Subtract  
- Multiply  
- Divide
- Gamma correction

These operate pixel-by-pixel with clipping to the valid range (0–255).

---

### **2. Grayscale Conversion**
Implemented in several modes:
- Standard grayscale (lominosity)
- Mean  
- Per-channel grayscale (R / G / B)

---

### **3. Image Enhancement Filters**
- **Mean (averaging) filter**  
- **Median filter**  
- **Gaussian blur**  
- **Sharpen filter**

Each filter is implemented using **manual convolution via NumPy**, without external helper libraries.

---

### **4. Sobel Edge Detection**
Supports Sobel masks at:
- **0°**
- **45°**
- **90°**
- **135°**

Multiple angles can be combined.
Only four kernels are needed — the remaining ones are 180° rotations differing only by sign.

---

### **5. Morphological Filters**

Binary morphology operations for shape analysis and noise removal:

- **Binary conversion** — Threshold-based conversion to binary image (0/255)
- **Dilation** — Expands bright regions (foreground)
- **Erosion** — Shrinks bright regions, removes noise
- **Opening** — Erosion followed by dilation (removes small bright spots)
- **Closing** — Dilation followed by erosion (fills small dark holes)

All morphological filters convert images to grayscale and apply binarization (threshold: 128 by default).
Structuring elements (SE) are square matrices (3×3, 5×5, etc.) filled with ones.

---

### **6. Custom Convolution**

The user can apply a **custom kernel of any valid odd size**
(e.g., 3×3, 5×5, 7×7…).

---
## ▶️ Running the Project

### **Demo Script**

Run the demo script to generate example outputs:
```bash
python Examples.py
```

### **Interactive CLI**

Run the CLI tool for interactive image processing:

```bash
python app.py
```

Once launched, you'll see a menu of available commands organized by category.

---

## 🛠️ CLI Usage Guide

### **Quick Start**

1. **Start the CLI**:
   ```bash
   python app.py
   ```

2. **Load an image**:
   ```
   > load example.jpg
   ```

3. **Apply filters**:
   ```
   > gaussian 1.5 5
   > sobel
   > save output.png
   ```

4. **Exit**:
   ```
   > exit
   ```

---

### **Command Categories**

#### **Global Commands**

| Command   | Arguments  | Description                             | Example           |
|-----------|------------|-----------------------------------------|-------------------|
| `load`    | `filename` | Load image from input directory         | `load img.jpg`    |
| `save`    | `filename` | Save current result to output directory | `save result.png` |
| `undo`    | —          | Undo last operation                     | `undo`            |
| `redo`    | —          | Redo previously undone operation        | `redo`            |
| `history` | —          | Show execution history                  | `history`         |
| `exit`    | —          | Exit the program                        | `exit`            |

#### **Configuration Commands**

| Command      | Arguments        | Description                                                | Example               |
|--------------|------------------|------------------------------------------------------------|-----------------------|
| `set_input`  | `path`           | Set input directory                                        | `set_input ./images`  |
| `set_output` | `path`           | Set output directory                                       | `set_output ./output` |
| `mode`       | `val [filename]` | Set save mode (0=Manual, 1=Auto-NewFile, 2=Auto-FixedFile) | `mode 1`              |
| `info`       | —                | Show current configuration                                 | `info`                |

#### **Enhancement Filters**

| Command    | Arguments              | Description             | Example          |
|------------|------------------------|-------------------------|------------------|
| `mean`     | `[size=3]`             | Apply mean (box) filter | `mean 5`         |
| `sharpen`  | —                      | Apply sharpen filter    | `sharpen`        |
| `gaussian` | `[sigma=1.0] [size=3]` | Apply Gaussian blur     | `gaussian 1.5 5` |
| `median`   | `[ksize=3]`            | Apply median filter     | `median 5`       |
| `sobel`    | `[angles=(0,90)]`      | Sobel edge detector     | `sobel 0,45,90`  |

#### **Arithmetic Operations**

| Command | Arguments | Description                                                       | Example         |
|---------|-----------|-------------------------------------------------------------------|-----------------|
| `add`   | `value`   | Add value to all pixels                                           | `add 50`        |
| `sub`   | `value`   | Subtract value from all pixels                                    | `sub 30`        |
| `mul`   | `value`   | Multiply image by value                                           | `mul 1.5`       |
| `div`   | `value`   | Divide image by value                                             | `div 2.0`       |
| `gamma` | `value`   | Apply gamma correction                                            | `gamma 1.2`     |
| `gray`  | `method`  | Convert to grayscale (methods: `standard`, `mean`, `r`, `g`, `b`) | `gray standard` |

#### **Morphological Filters** ✨ NEW

| Command  | Arguments         | Description                                        | Example      |
|----------|-------------------|----------------------------------------------------|--------------|
| `binary` | `[threshold=128]` | Convert to binary (0 or 255)                       | `binary 128` |
| `dilate` | `[size=3]`        | Morphological dilation (expands bright regions)    | `dilate 3`   |
| `erode`  | `[size=3]`        | Morphological erosion (shrinks bright regions)     | `erode 3`    |
| `open`   | `[size=3]`        | Morphological opening (removes small bright spots) | `open 5`     |
| `close`  | `[size=3]`        | Morphological closing (fills small dark holes)     | `close 5`    |

**Note**: All morphological operations automatically convert images to grayscale and apply binarization. The `size`
parameter defines the structuring element dimensions (e.g., `size=3` creates a 3×3 SE).

---

### **Usage Examples**

#### Example 1: Basic Image Processing Workflow

```
> load photo.jpg
> gaussian 1.0 5
> sharpen
> save enhanced.png
```

#### Example 2: Edge Detection

```
> load document.jpg
> gray standard
> sobel 0,90
> save edges.png
```

#### Example 3: Morphological Operations

```
> load binary_image.png
> binary 128
> dilate 3
> save dilated.png
> undo
> erode 3
> save eroded.png
```

#### Example 4: Noise Removal with Opening

```
> load noisy.jpg
> binary 150
> open 5
> save cleaned.png
```

#### Example 5: Hole Filling with Closing

```
> load text.png
> binary 120
> close 3
> save filled.png
```

---

### **Argument Syntax**

Commands support two argument styles:

1. **Positional**: `gaussian 1.5 5`
2. **Named**: `gaussian sigma=1.5 size=5`
3. **Mixed**: `gaussian 1.5 size=5`

Optional arguments have default values shown in square brackets (e.g., `[size=3]`).

---

### **Save Modes**

Configure automatic saving behavior:

- **Mode 0 (Manual)**: Images are saved only when you run `save filename`
- **Mode 1 (Auto-NewFile)**: After each operation, automatically saves with a generated filename
- **Mode 2 (Auto-FixedFile)**: After each operation, automatically overwrites a specific filename

Set mode with: `mode <0|1|2> [filename]`

Example:

```
> mode 1              # Auto-save with generated names
> mode 2 output.png   # Auto-save to output.png
> mode 0              # Back to manual mode
```

---

## 🖼 Example Results

### Input Image  
`input/example.jpg`  
![Input](input/example.jpg)

---

### Example Outputs

#### Median Filter  
`output/examples/median.jpg`  
![Median](output/examples/median.jpg)

#### Gaussian Blur  
`output/examples/gaussian.jpg`  
![Gaussian](output/examples/gaussian.jpg)

#### Sharpen Filter  
`output/examples/sharpen.jpg`  
![Sharpen](output/examples/sharpen.jpg)

#### Sobel (0°, 45°, 90°, 135° combined)  
`output/examples/sobel_all.jpg`  
![Sobel](output/examples/sobel_all.jpg)

#### Arithmetic: Add  
`output/examples/add.jpg`  
![Add](output/examples/add.jpg)

## 🧩 Dependencies

Only:
- numpy
- Pillow
- tqdm (used for progress bar in `Examples.py`)

The project intentionally avoids OpenCV or SciPy to preserve the educational nature.

## 🎓 Purpose
This project is designed to help understand:
- how convolution works step-by-step
- how filters modify pixel matrices
- how Sobel detects gradients
- how pixel-wise operations affect brightness and contrast
- how grayscale transforms can be implemented manually

It is not a production-ready image-processing library — but a clear and well-structured educational implementation.

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this code for educational or personal purposes.

See the full license text in the [LICENSE](LICENSE) file.
