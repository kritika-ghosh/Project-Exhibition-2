"""
NLI-Based File Sorting System - Clean Version
Tabs: Dashboard, Homepage, Automated
Working HITL with visible submit button
Includes Poppler from: C:\Program Files\poppler-25.12.0\Library\bin
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the prediction function
try:
    from integrated import predict_against_folders, update_model, embedder
    INTEGRATED_AVAILABLE = True
except ImportError as e:
    print(f"Error importing integrated module: {e}")
    INTEGRATED_AVAILABLE = False

# For PDF text extraction
try:
    import PyPDF2
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False

# =========================
# OCR SETUP WITH SPECIFIC POPPLER PATH
# =========================

# Set Poppler path - YOUR SPECIFIC PATH
POPPLER_PATH = r"C:\Program Files\poppler-25.12.0\Library\bin"

# Check if Poppler exists at the specified path
if os.path.exists(POPPLER_PATH):
    print(f"✓ Poppler found at: {POPPLER_PATH}")
    POPPLER_AVAILABLE = True
else:
    print(f"⚠️ Poppler not found at: {POPPLER_PATH}")
    print("   Please verify the path is correct")
    POPPLER_AVAILABLE = False

# Try to import OCR libraries
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
    
    # Configure pdf2image to use our Poppler path
    if POPPLER_AVAILABLE:
        pdf2image.poppler_path = POPPLER_PATH
        
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"⚠️ OCR libraries not fully installed: {e}")
    print("   Install with: pip install pytesseract pdf2image pillow")

# Try to find Tesseract
def find_tesseract():
    """Find tesseract executable on system"""
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME')),
        r"C:\Tesseract-OCR\tesseract.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Check if it's in PATH
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        return tesseract_path
        
    return None

TESSERACT_PATH = find_tesseract()
if TESSERACT_PATH and OCR_AVAILABLE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"✓ Tesseract found at: {TESSERACT_PATH}")
    TESSERACT_AVAILABLE = True
else:
    TESSERACT_AVAILABLE = False
    print("⚠️ Tesseract not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki")

# =========================
# CONFIG
# =========================

CONFIG_FILE = "nli_sorter_config.json"

DEFAULT_CONFIG = {
    "target_folders": {},
    "thresholds": {
        "min_confidence": 0.35
    }
}

# =========================
# TEXT EXTRACTION WITH OCR & POPPLER
# =========================

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file - works with screenshots via OCR (uses your Poppler path)"""
    text = ""
    
    # Method 1: Try PyPDF2 for selectable text
    if PDF_EXTRACTION_AVAILABLE:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
            
            if len(text.strip()) > 200:
                text = ' '.join(text.split())
                return text[:3000]
        except Exception as e:
            print(f"  PyPDF2 failed: {e}")
    
    # Method 2: Try OCR for scanned PDFs/screenshots (using your Poppler path)
    if OCR_AVAILABLE and TESSERACT_AVAILABLE and POPPLER_AVAILABLE and len(text.strip()) < 200:
        try:
            print(f"  Attempting OCR on {os.path.basename(pdf_path)}...")
            
            # Convert PDF to images using your Poppler path
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=200,
                poppler_path=POPPLER_PATH  # Explicitly use your Poppler path
            )
            
            print(f"  Converted {len(images)} pages to images")
            
            for i, img in enumerate(images):
                # Preprocess for better OCR
                if img.mode != 'L':
                    img = img.convert('L')
                # Increase contrast for better text detection
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0)
                
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text:
                    text += ocr_text + "\n"
                    print(f"  Page {i+1}: OCR extracted {len(ocr_text)} chars")
            
            if text.strip():
                print(f"  ✓ OCR extracted total {len(text)} characters")
                text = ' '.join(text.split())
                return text[:3000]
            else:
                print(f"  ⚠️ OCR produced no text. Check if the PDF contains readable text.")
                
        except Exception as e:
            error_msg = str(e).lower()
            if 'poppler' in error_msg:
                print(f"  ❌ OCR failed: Poppler not found at {POPPLER_PATH}")
                print(f"     Please verify the path: C:\\Program Files\\poppler-25.12.0\\Library\\bin")
            else:
                print(f"  ❌ OCR failed: {e}")
    
    # Method 3: Try direct image OCR if it's an image file
    if not text and pdf_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            print(f"  Attempting OCR on image file...")
            img = Image.open(pdf_path)
            if img.mode != 'L':
                img = img.convert('L')
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            text = pytesseract.image_to_string(img)
            print(f"  Image OCR extracted {len(text)} chars")
            text = ' '.join(text.split())
            return text[:3000]
        except Exception as e:
            print(f"  Image OCR failed: {e}")
    
    return text[:3000] if text else ""

def extract_text_from_txt(txt_path):
    """Extract text from TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text[:3000]
    except Exception as e:
        return ""

def extract_text_from_file(file_path):
    """Extract text from various file types"""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        return ""

# =========================
# HITL DIALOG - WITH WORKING SUBMIT BUTTON
# =========================

class HITLDialog(tk.Toplevel):
    """Human-in-the-loop dialog for manual classification"""
    
    def __init__(self, parent, filename, folders, file_path, text_content):
        super().__init__(parent)
        self.title("Manual Classification Required")
        self.configure(bg="#0D0F14")
        self.geometry("700x600")
        self.resizable(False, False)
        self.selected_folder = None
        self.folders = folders
        
        # Make it modal
        self.transient(parent)
        self.grab_set()
        
        # Build UI
        self._build_ui(filename, text_content)
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
        
        # Wait for user input
        self.wait_window()
        
    def _build_ui(self, filename, text_content):
        # Main container with padding
        main = tk.Frame(self, bg="#0D0F14")
        main.pack(fill="both", expand=True, padx=25, pady=25)
        
        # Header with warning icon
        header_frame = tk.Frame(main, bg="#0D0F14")
        header_frame.pack(fill="x", pady=(0, 20))
        
        tk.Label(header_frame, text="⚠️", fg="#FFB547", bg="#0D0F14",
                 font=("Segoe UI", 32)).pack(side="left", padx=(0, 15))
        
        header_text = tk.Frame(header_frame, bg="#0D0F14")
        header_text.pack(side="left", fill="x", expand=True)
        
        tk.Label(header_text, text="MANUAL CLASSIFICATION REQUIRED", 
                 fg="#FFB547", bg="#0D0F14", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(header_text, text="The system could not confidently classify this file.", 
                 fg="#8B90A8", bg="#0D0F14", font=("Segoe UI", 10)).pack(anchor="w", pady=(5, 0))
        
        # File information card
        file_card = tk.Frame(main, bg="#1A1D27", relief="flat", bd=0)
        file_card.pack(fill="x", pady=(0, 15))
        
        tk.Label(file_card, text="📄", fg="#6C63FF", bg="#1A1D27",
                 font=("Segoe UI", 20)).pack(side="left", padx=15, pady=15)
        tk.Label(file_card, text=filename, fg="#E8EAF6", bg="#1A1D27",
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=10, pady=15)
        
        # Text preview card
        preview_card = tk.Frame(main, bg="#1A1D27", relief="flat", bd=0)
        preview_card.pack(fill="x", pady=(0, 15))
        
        tk.Label(preview_card, text="📝 Text Preview (first 300 characters):", 
                 fg="#4A4F6A", bg="#1A1D27", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15, pady=(10, 5))
        
        preview = text_content[:300] + "..." if len(text_content) > 300 else text_content
        text_widget = tk.Text(preview_card, bg="#0D0F14", fg="#8B90A8",
                               font=("Segoe UI", 9), wrap="word", height=5)
        text_widget.pack(fill="both", padx=15, pady=(0, 15))
        text_widget.insert("1.0", preview)
        text_widget.config(state="disabled")
        
        # Folder selection label
        tk.Label(main, text="SELECT THE CORRECT FOLDER:", 
                 fg="#4A4F6A", bg="#0D0F14", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Scrollable folder list
        list_container = tk.Frame(main, bg="#111420", relief="flat", bd=1, highlightbackground="#2A2D3E", highlightthickness=1)
        list_container.pack(fill="both", expand=True, pady=(0, 20))
        
        canvas = tk.Canvas(list_container, bg="#111420", highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg="#111420")
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.folder_var = tk.StringVar()
        
        # Create radio buttons for each folder
        for i, folder in enumerate(self.folders):
            rb_frame = tk.Frame(scrollable, bg="#111420")
            rb_frame.pack(fill="x", padx=15, pady=8)
            
            rb = tk.Radiobutton(
                rb_frame, 
                text=f"📁  {folder}",
                variable=self.folder_var, 
                value=folder,
                fg="#E8EAF6", 
                bg="#111420",
                selectcolor="#3D3880",
                activebackground="#1A1D27",
                activeforeground="#E8EAF6",
                font=("Segoe UI", 12), 
                anchor="w",
                cursor="hand2"
            )
            rb.pack(side="left", fill="x", expand=True)
            
            # Select first folder by default
            if i == 0:
                self.folder_var.set(folder)
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        # Button frame - with prominent submit button
        btn_frame = tk.Frame(main, bg="#0D0F14")
        btn_frame.pack(fill="x", pady=(10, 0))
        
        # Skip button (secondary)
        skip_btn = tk.Button(
            btn_frame, 
            text="Skip File", 
            command=self._skip,
            bg="#2A2D3E", 
            fg="#8B90A8", 
            font=("Segoe UI", 11),
            padx=30, 
            pady=10, 
            cursor="hand2",
            relief="flat"
        )
        skip_btn.pack(side="left", padx=(0, 15))
        
        # Submit button - BIG and PROMINENT
        submit_btn = tk.Button(
            btn_frame, 
            text="✓ SUBMIT & MOVE FILE", 
            command=self._confirm,
            bg="#6C63FF", 
            fg="white", 
            font=("Segoe UI", 13, "bold"),
            padx=40, 
            pady=12, 
            cursor="hand2",
            relief="flat"
        )
        submit_btn.pack(side="right")
        
        # Bind Enter key to submit
        self.bind('<Return>', lambda e: self._confirm())
        
        # Set focus to submit button
        submit_btn.focus_set()
        
    def _confirm(self):
        """Confirm and move file"""
        self.selected_folder = self.folder_var.get()
        self.destroy()
        
    def _skip(self):
        """Skip this file"""
        self.selected_folder = None
        self.destroy()

# =========================
# FILE PROCESSOR
# =========================

class FileProcessor:
    """Process files using NLI classification"""
    
    def __init__(self, config):
        self.config = config
        
    def process_file(self, file_path, folder_names):
        """Process a file and return prediction"""
        try:
            print(f"\nProcessing: {os.path.basename(file_path)}")
            
            # Extract text
            text = extract_text_from_file(file_path)
            
            if not text or len(text.strip()) < 50:
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": f"Insufficient text (length: {len(text)}). Install Tesseract and Poppler for screenshot support.",
                    "text": text
                }
            
            print(f"  Extracted {len(text)} chars of text")
            
            # Get prediction
            if INTEGRATED_AVAILABLE and folder_names:
                prediction, confidence, source, needs_hitl = predict_against_folders(
                    text=text,
                    folder_names=folder_names,
                    use_hierarchy=True,
                    use_ssp=True,
                    use_dlts=True
                )
                
                print(f"  Prediction: {prediction}")
                print(f"  Confidence: {confidence:.3f}")
                
                # Ensure prediction is a valid folder
                if prediction and prediction not in folder_names:
                    prediction = self._find_closest_match(prediction, folder_names)
                    print(f"  Adjusted to: {prediction}")
                
                return {
                    "file_path": file_path,
                    "success": True,
                    "prediction": prediction,
                    "confidence": confidence,
                    "source": source,
                    "needs_hitl": needs_hitl or confidence < self.config["thresholds"]["min_confidence"],
                    "text": text
                }
            else:
                return {
                    "file_path": file_path,
                    "success": False,
                    "error": "Classification not available",
                    "text": text
                }
                
        except Exception as e:
            print(f"  Error: {e}")
            return {
                "file_path": file_path,
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    def _find_closest_match(self, prediction, folder_names):
        """Find closest matching folder name"""
        prediction_lower = prediction.lower()
        
        for folder in folder_names:
            if folder.lower() == prediction_lower:
                return folder
        
        # Partial match
        for folder in folder_names:
            if prediction_lower in folder.lower() or folder.lower() in prediction_lower:
                return folder
        
        return folder_names[0] if folder_names else prediction

# =========================
# GUI STYLES
# =========================

BG_BASE = "#0D0F14"
BG_SURFACE = "#13161E"
BG_CARD = "#1A1D27"
BG_INPUT = "#111420"
BORDER = "#2A2D3E"
ACCENT = "#6C63FF"
SUCCESS = "#3DDC84"
WARN = "#FFB547"
DANGER = "#FF5C5C"
TEXT_PRI = "#E8EAF6"
TEXT_SEC = "#8B90A8"
TEXT_HINT = "#4A4F6A"

# =========================
# MAIN APPLICATION
# =========================

class NLIApplication(tk.Tk):
    """Main GUI Application"""
    
    def __init__(self):
        super().__init__()
        self.title("NLI File Sorter - Intelligent Document Organizer")
        self.geometry("1100x700")
        self.minsize(1000, 650)
        self.configure(bg=BG_BASE)
        
        # Load config
        self.config = self._load_config()
        
        # Initialize components
        self.processor = FileProcessor(self.config)
        
        # Tracking
        self.sort_results = []
        
        # Build UI
        self._build()
        
    def _load_config(self):
        """Load configuration from file"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    if "target_folders" not in config:
                        config["target_folders"] = {}
                    return config
            except:
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
            
    def _build(self):
        """Build the GUI"""
        # Main container
        main_container = tk.Frame(self, bg=BG_BASE)
        main_container.pack(fill="both", expand=True)
        
        # Sidebar
        sidebar = tk.Frame(main_container, bg=BG_SURFACE, width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        self._build_sidebar(sidebar)
        
        # Content area
        self.content_area = tk.Frame(main_container, bg=BG_BASE)
        self.content_area.pack(side="left", fill="both", expand=True)
        
        # Create panels
        self.panels = {}
        self._create_panels()
        
        # Show homepage by default
        self._show_panel("homepage")
        
    def _build_sidebar(self, sidebar):
        """Build sidebar navigation"""
        # Logo
        logo_frame = tk.Frame(sidebar, bg=BG_SURFACE)
        logo_frame.pack(fill="x", padx=20, pady=(25, 20))
        
        tk.Label(logo_frame, text="NLI SORT", fg=ACCENT, bg=BG_SURFACE,
                 font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(logo_frame, text="Intelligent File Sorter",
                 fg=TEXT_HINT, bg=BG_SURFACE, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))
        
        # Separator
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)
        
        # Navigation buttons
        nav_items = [
            ("🏠 Homepage", "homepage"),
            ("📊 Dashboard", "dashboard"),
            ("🤖 Automated Sorting", "automated")
        ]
        
        self.nav_buttons = {}
        for label, panel_id in nav_items:
            btn = tk.Button(
                sidebar, text=f"  {label}",
                bg=BG_SURFACE, fg=TEXT_SEC,
                font=("Segoe UI", 11), anchor="w",
                relief="flat", cursor="hand2", bd=0,
                padx=15, pady=12,
                activebackground=BG_CARD,
                activeforeground=TEXT_PRI,
                command=lambda p=panel_id: self._show_panel(p)
            )
            btn.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[panel_id] = btn
        
        # Status indicator at bottom
        tk.Frame(sidebar, bg=BG_SURFACE).pack(fill="y", expand=True)
        
        status_frame = tk.Frame(sidebar, bg=BG_SURFACE)
        status_frame.pack(fill="x", padx=20, pady=15)
        
        self.status_indicator = tk.Label(status_frame, text="●", fg=SUCCESS, bg=BG_SURFACE,
                                         font=("Segoe UI", 10))
        self.status_indicator.pack(side="left")
        self.status_text = tk.Label(status_frame, text="  Ready",
                                    fg=TEXT_HINT, bg=BG_SURFACE, font=("Segoe UI", 9))
        self.status_text.pack(side="left")
        
    def _create_panels(self):
        """Create all panels"""
        # Homepage Panel
        homepage = tk.Frame(self.content_area, bg=BG_BASE)
        self._build_homepage(homepage)
        self.panels["homepage"] = homepage
        
        # Dashboard Panel
        dashboard = tk.Frame(self.content_area, bg=BG_BASE)
        self._build_dashboard(dashboard)
        self.panels["dashboard"] = dashboard
        
        # Automated Panel
        automated = tk.Frame(self.content_area, bg=BG_BASE)
        self._build_automated_panel(automated)
        self.panels["automated"] = automated
        
    def _build_homepage(self, parent):
        """Build beautiful homepage with instructions including KNN"""
        # Scrollable container
        canvas = tk.Canvas(parent, bg=BG_BASE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg=BG_BASE)
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Main content
        main = scrollable
        
        # Hero section
        hero = tk.Frame(main, bg=BG_BASE)
        hero.pack(fill="x", pady=(40, 30))
        
        tk.Label(hero, text="🤖", fg=ACCENT, bg=BG_BASE,
                 font=("Segoe UI", 64)).pack()
        tk.Label(hero, text="NLI File Sorter", fg=TEXT_PRI, bg=BG_BASE,
                 font=("Segoe UI", 32, "bold")).pack(pady=(10, 5))
        tk.Label(hero, text="Intelligent Document Organization Using AI", 
                 fg=TEXT_SEC, bg=BG_BASE, font=("Segoe UI", 14)).pack()
        
        # What is this?
        what_card = tk.Frame(main, bg=BG_CARD, relief="flat", bd=0)
        what_card.pack(fill="x", padx=40, pady=15)
        
        tk.Label(what_card, text="📌 WHAT IS THIS?", fg=ACCENT, bg=BG_CARD,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        tk.Label(what_card, 
                 text="NLI File Sorter is an AI-powered tool that automatically organizes your documents\n"
                      "into appropriate folders based on their content. It uses a hybrid approach combining\n"
                      "KNN (K-Nearest Neighbors) and NLI (Natural Language Inference) to understand\n"
                      "what each document is about and moves it to the correct location.",
                 fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 11), justify="left").pack(anchor="w", padx=20, pady=(0, 15))
        
        # How it works - Updated with KNN and NLI
        how_card = tk.Frame(main, bg=BG_CARD, relief="flat", bd=0)
        how_card.pack(fill="x", padx=40, pady=15)
        
        tk.Label(how_card, text="⚙️ HOW IT WORKS", fg=ACCENT, bg=BG_CARD,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        
        steps = [
            ("1", "Extract Text", "Reads text from PDFs and images using OCR technology (requires Poppler)"),
            ("2", "KNN Analysis", "Uses K-Nearest Neighbors to find similar documents from past classifications"),
            ("3", "NLI Verification", "Validates the match using Natural Language Inference for accuracy"),
            ("4", "Confidence Scoring", "Calculates confidence score based on both KNN and NLI results"),
            ("5", "Auto or Manual", "High confidence → Auto-sort | Low confidence → Ask for help (HITL)"),
            ("6", "Continuous Learning", "Every sort (auto or manual) improves the KNN model for future files")
        ]
        
        for num, title, desc in steps:
            step_frame = tk.Frame(how_card, bg=BG_CARD)
            step_frame.pack(fill="x", padx=20, pady=8)
            
            tk.Label(step_frame, text=num, fg=ACCENT, bg=BG_CARD,
                     font=("Segoe UI", 16, "bold"), width=3).pack(side="left")
            
            step_text = tk.Frame(step_frame, bg=BG_CARD)
            step_text.pack(side="left", fill="x", expand=True)
            
            tk.Label(step_text, text=title, fg=TEXT_PRI, bg=BG_CARD,
                     font=("Segoe UI", 12, "bold")).pack(anchor="w")
            tk.Label(step_text, text=desc, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 10)).pack(anchor="w")
        
        tk.Frame(how_card, height=10, bg=BG_CARD).pack()
        
        # Technology section - Explain KNN and NLI
        tech_card = tk.Frame(main, bg=BG_CARD, relief="flat", bd=0)
        tech_card.pack(fill="x", padx=40, pady=15)
        
        tk.Label(tech_card, text="🔬 TECHNOLOGY BEHIND THE SYSTEM", fg=ACCENT, bg=BG_CARD,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        
        techs = [
            ("🧠 KNN (K-Nearest Neighbors)", 
             "Remembers past classifications. When a new file arrives, KNN finds the most similar documents\n"
             "from its memory and suggests their categories. Gets smarter with every file you sort."),
            ("🎯 NLI (Natural Language Inference)", 
             "Understands document content at a semantic level. Verifies if the document actually belongs\n"
             "to the suggested category by analyzing the relationship between the text and category name."),
            ("🤝 Hybrid Approach", 
             "Combines KNN's memory-based learning with NLI's deep understanding for more accurate results.\n"
             "KNN provides speed and memory, NLI provides verification and handling of new categories."),
            ("📚 Continuous Learning", 
             "Every time you sort a file (automatically or manually), the system learns from that decision.\n"
             "The KNN model updates immediately, making future classifications more accurate."),
            ("👤 Human-in-the-Loop (HITL)", 
             "When confidence is low, the system asks for human input. This ensures accuracy while still\n"
             "benefiting from automation. Each manual sort trains the model for similar future files.")
        ]
        
        for title, desc in techs:
            tech_frame = tk.Frame(tech_card, bg=BG_CARD)
            tech_frame.pack(fill="x", padx=20, pady=8)
            
            tk.Label(tech_frame, text=title, fg=TEXT_PRI, bg=BG_CARD,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w")
            tk.Label(tech_frame, text=desc, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 10), justify="left").pack(anchor="w", padx=(20, 0))
        
        tk.Frame(tech_card, height=10, bg=BG_CARD).pack()
        
        # How to use
        use_card = tk.Frame(main, bg=BG_CARD, relief="flat", bd=0)
        use_card.pack(fill="x", padx=40, pady=15)
        
        tk.Label(use_card, text="📖 HOW TO USE", fg=ACCENT, bg=BG_CARD,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        
        instructions = [
            "1. Go to the 'Automated Sorting' tab from the sidebar",
            "2. Click 'Browse' next to 'Source Folder' and select a folder with unsorted files",
            "3. Click 'Browse' next to 'Target Folder' and select a folder with subfolders as categories",
            "4. Click 'RUN SORTING' to start the automated process",
            "5. Watch as KNN and NLI analyze each file",
            "6. If the system is unsure (low confidence), a popup will ask you to manually select the correct folder",
            "7. Each manual selection trains the KNN model for future accuracy",
            "8. Check the Dashboard to see statistics and recent activity"
        ]
        
        for instruction in instructions:
            tk.Label(use_card, text=instruction, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 11)).pack(anchor="w", padx=20, pady=4)
        
        tk.Frame(use_card, height=10, bg=BG_CARD).pack()
        
        # Requirements section
        req_card = tk.Frame(main, bg=BG_CARD, relief="flat", bd=0)
        req_card.pack(fill="x", padx=40, pady=15)
        
        tk.Label(req_card, text="💻 REQUIREMENTS", fg=ACCENT, bg=BG_CARD,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        
        requirements = [
            "✅ Python 3.8+ with required packages (torch, transformers, sentence-transformers, scikit-learn)",
            "✅ Tesseract OCR (for text extraction from scanned PDFs and screenshots)",
            "✅ Poppler (for PDF to image conversion) - Configured at: C:\\Program Files\\poppler-25.12.0\\Library\\bin",
            "✅ PDF/TXT files to sort",
            "✅ Target folder with subfolders as categories"
        ]
        
        for req in requirements:
            tk.Label(req_card, text=req, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 11)).pack(anchor="w", padx=20, pady=4)
        
        tk.Frame(req_card, height=15, bg=BG_CARD).pack()
        
        # Features
        features_card = tk.Frame(main, bg=BG_CARD, relief="flat", bd=0)
        features_card.pack(fill="x", padx=40, pady=15)
        
        tk.Label(features_card, text="✨ FEATURES", fg=ACCENT, bg=BG_CARD,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        
        features = [
            "✅ Hybrid KNN + NLI classification for accurate results",
            "✅ Supports PDF and TXT files",
            "✅ OCR support for scanned documents and screenshots (Poppler configured)",
            "✅ Automatic folder matching based on content analysis",
            "✅ Human-in-the-loop for uncertain cases (HITL dialog)",
            "✅ Continuous learning from your sorting decisions",
            "✅ Batch processing for multiple files at once",
            "✅ Real-time progress tracking and detailed logs",
            "✅ Persistent KNN model that improves over time"
        ]
        
        for feature in features:
            tk.Label(features_card, text=feature, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 11)).pack(anchor="w", padx=20, pady=4)
        
        tk.Frame(features_card, height=15, bg=BG_CARD).pack()
        
        # Footer
        footer = tk.Frame(main, bg=BG_BASE)
        footer.pack(fill="x", pady=30)
        
        tk.Label(footer, text="Ready to organize your files? → Go to the Automated Sorting tab!",
                 fg=TEXT_HINT, bg=BG_BASE, font=("Segoe UI", 11)).pack()
        
    def _build_dashboard(self, parent):
        """Build dashboard panel with statistics"""
        # Stats cards
        stats_frame = tk.Frame(parent, bg=BG_BASE)
        stats_frame.pack(fill="x", pady=(30, 30), padx=30)
        
        stats = [
            ("📊", "Files Sorted", "0"),
            ("✓", "Auto Sorted", "0"),
            ("👤", "Manual (HITL)", "0"),
            ("✗", "Failed", "0"),
            ("🧠", "KNN Memory", "0")
        ]
        
        self.stat_labels = {}
        for icon, label, value in stats:
            card = tk.Frame(stats_frame, bg=BG_CARD, relief="flat", bd=0)
            card.pack(side="left", expand=True, fill="x", padx=8, ipady=15)
            
            tk.Label(card, text=icon, fg=ACCENT, bg=BG_CARD,
                     font=("Segoe UI", 28)).pack(anchor="w", padx=15, pady=(10, 0))
            
            self.stat_labels[label] = tk.Label(card, text=value, fg=TEXT_PRI, bg=BG_CARD,
                                                font=("Segoe UI", 28, "bold"))
            self.stat_labels[label].pack(anchor="w", padx=15, pady=(5, 0))
            
            tk.Label(card, text=label, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 10)).pack(anchor="w", padx=15, pady=(0, 10))
        
        # Recent activity
        activity_frame = tk.Frame(parent, bg=BG_BASE)
        activity_frame.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        tk.Label(activity_frame, text="📋 Recent Activity", fg=TEXT_PRI, bg=BG_BASE,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Activity list with scrollbar
        list_container = tk.Frame(activity_frame, bg=BG_CARD, relief="flat", bd=0)
        list_container.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        
        self.activity_list = tk.Listbox(list_container, bg=BG_CARD, fg=TEXT_SEC,
                                         font=("Segoe UI", 10), relief="flat",
                                         selectbackground=BG_INPUT,
                                         yscrollcommand=scrollbar.set)
        self.activity_list.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.activity_list.yview)
        
        # Add welcome message
        self.activity_list.insert(0, "  🎉 Welcome to NLI File Sorter!")
        self.activity_list.insert(1, "  🧠 Using KNN + NLI hybrid classification")
        self.activity_list.insert(2, "  📁 Go to Automated Sorting tab to start organizing your files")
        
        # Initialize KNN memory stat
        self._update_knn_memory_stat()
        
    def _build_automated_panel(self, parent):
        """Build automated sorting panel"""
        main_frame = tk.Frame(parent, bg=BG_BASE)
        main_frame.pack(fill="both", expand=True, padx=40, pady=30)
        
        # Title
        tk.Label(main_frame, text="Automated Batch Sorting", fg=TEXT_PRI, bg=BG_BASE,
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", pady=(0, 5))
        tk.Label(main_frame, text="KNN + NLI hybrid classification - The system will ask for help when needed",
                 fg=TEXT_HINT, bg=BG_BASE, font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 25))
        
        # Source folder selection
        source_frame = tk.Frame(main_frame, bg=BG_CARD, relief="flat", bd=0)
        source_frame.pack(fill="x", pady=8)
        
        source_inner = tk.Frame(source_frame, bg=BG_CARD)
        source_inner.pack(fill="x", padx=20, pady=15)
        
        tk.Label(source_inner, text="📂 Source Folder (Unsorted):", fg=TEXT_PRI, bg=BG_CARD,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        
        source_select = tk.Frame(source_inner, bg=BG_CARD)
        source_select.pack(fill="x", pady=10)
        
        self.source_folder_var = tk.StringVar()
        source_entry = tk.Entry(source_select, textvariable=self.source_folder_var,
                                 bg=BG_INPUT, fg=TEXT_PRI, font=("Segoe UI", 10),
                                 width=60, relief="flat")
        source_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        tk.Button(source_select, text="Browse", command=self._browse_source_folder,
                  bg=BG_INPUT, fg=TEXT_SEC, font=("Segoe UI", 10),
                  cursor="hand2", padx=20, pady=6, relief="flat").pack(side="left")
        
        # Target folder selection
        target_frame = tk.Frame(main_frame, bg=BG_CARD, relief="flat", bd=0)
        target_frame.pack(fill="x", pady=8)
        
        target_inner = tk.Frame(target_frame, bg=BG_CARD)
        target_inner.pack(fill="x", padx=20, pady=15)
        
        tk.Label(target_inner, text="🎯 Target Folder (with subfolders):", fg=TEXT_PRI, bg=BG_CARD,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(target_inner, text="The system will only sort into existing subfolders",
                 fg=TEXT_HINT, bg=BG_CARD, font=("Segoe UI", 9)).pack(anchor="w", pady=(5, 10))
        
        target_select = tk.Frame(target_inner, bg=BG_CARD)
        target_select.pack(fill="x", pady=10)
        
        self.target_folder_var = tk.StringVar()
        target_entry = tk.Entry(target_select, textvariable=self.target_folder_var,
                                 bg=BG_INPUT, fg=TEXT_PRI, font=("Segoe UI", 10),
                                 width=60, relief="flat")
        target_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        tk.Button(target_select, text="Browse", command=self._browse_target_folder,
                  bg=BG_INPUT, fg=TEXT_SEC, font=("Segoe UI", 10),
                  cursor="hand2", padx=20, pady=6, relief="flat").pack(side="left")
        
        # Info about KNN
        info_frame = tk.Frame(main_frame, bg=BG_CARD, relief="flat", bd=0)
        info_frame.pack(fill="x", pady=8)
        
        info_inner = tk.Frame(info_frame, bg=BG_CARD)
        info_inner.pack(fill="x", padx=20, pady=10)
        
        tk.Label(info_inner, text="🧠 How KNN + NLI Works:", fg=ACCENT, bg=BG_CARD,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(info_inner, text="• KNN remembers past classifications and finds similar documents\n• NLI verifies the match by understanding document content\n• High confidence = Auto-sort | Low confidence = Ask for help\n• Every sort (auto or manual) improves the KNN model",
                 fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 9), justify="left").pack(anchor="w", pady=(5, 0))
        
        # OCR Info
        if POPPLER_AVAILABLE:
            tk.Label(info_inner, text=f"✓ Poppler configured at: {POPPLER_PATH}", 
                     fg=SUCCESS, bg=BG_CARD, font=("Segoe UI", 8)).pack(anchor="w", pady=(5, 0))
        else:
            tk.Label(info_inner, text=f"⚠️ Poppler not found at: {POPPLER_PATH}", 
                     fg=WARN, bg=BG_CARD, font=("Segoe UI", 8)).pack(anchor="w", pady=(5, 0))
        
        # Run button
        btn_frame = tk.Frame(main_frame, bg=BG_BASE)
        btn_frame.pack(fill="x", pady=25)
        
        self.run_sort_btn = tk.Button(btn_frame, text="▶ RUN SORTING", command=self._run_automated_sort,
                                       bg=SUCCESS, fg="white", font=("Segoe UI", 13, "bold"),
                                       padx=40, pady=12, cursor="hand2", relief="flat")
        self.run_sort_btn.pack()
        
        # Progress area
        progress_frame = tk.Frame(main_frame, bg=BG_CARD, relief="flat", bd=0)
        progress_frame.pack(fill="both", expand=True, pady=15)
        
        progress_inner = tk.Frame(progress_frame, bg=BG_CARD)
        progress_inner.pack(fill="both", expand=True, padx=20, pady=15)
        
        tk.Label(progress_inner, text="Sorting Progress", fg=TEXT_PRI, bg=BG_CARD,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_inner, variable=self.progress_var,
                                             maximum=100, style="TProgressbar")
        self.progress_bar.pack(fill="x", pady=10)
        
        # Status label
        self.sort_status = tk.Label(progress_inner, text="Ready to start", fg=TEXT_SEC, bg=BG_CARD,
                                     font=("Segoe UI", 10))
        self.sort_status.pack(anchor="w", pady=5)
        
        # Results log
        log_label = tk.Label(progress_inner, text="Activity Log:", fg=TEXT_HINT, bg=BG_CARD,
                              font=("Segoe UI", 10, "bold"))
        log_label.pack(anchor="w", pady=(10, 5))
        
        self.sort_log = tk.Text(progress_inner, bg=BG_INPUT, fg=TEXT_SEC,
                                 font=("Segoe UI", 9), wrap="word", height=10,
                                 relief="flat")
        self.sort_log.pack(fill="both", expand=True, pady=(0, 5))
        
        scrollbar = ttk.Scrollbar(self.sort_log, command=self.sort_log.yview)
        self.sort_log.configure(yscrollcommand=scrollbar.set)
        
    def _show_panel(self, panel_id):
        """Show selected panel"""
        for pid, btn in self.nav_buttons.items():
            if pid == panel_id:
                btn.config(fg=ACCENT, bg=BG_CARD, font=("Segoe UI", 11, "bold"))
            else:
                btn.config(fg=TEXT_SEC, bg=BG_SURFACE, font=("Segoe UI", 11))
        
        for pid, panel in self.panels.items():
            if pid == panel_id:
                panel.pack(fill="both", expand=True)
            else:
                panel.pack_forget()
                
    def _browse_source_folder(self):
        """Browse for source folder"""
        path = filedialog.askdirectory(title="Select Source Folder with Unsorted Files")
        if path:
            self.source_folder_var.set(path)
            
    def _browse_target_folder(self):
        """Browse for target folder"""
        path = filedialog.askdirectory(title="Select Target Folder with Subfolders")
        if path:
            self.target_folder_var.set(path)
            
    def _run_automated_sort(self):
        """Run automated sorting with HITL support"""
        source = self.source_folder_var.get().strip()
        target = self.target_folder_var.get().strip()
        
        if not source or not target:
            messagebox.showwarning("Missing Folders", "Please select both source and target folders")
            return
            
        if not os.path.exists(source):
            messagebox.showerror("Error", "Source folder does not exist")
            return
            
        if not os.path.exists(target):
            messagebox.showerror("Error", "Target folder does not exist")
            return
        
        # Get subfolders from target
        subfolders = []
        for item in os.listdir(target):
            item_path = os.path.join(target, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
        
        if not subfolders:
            messagebox.showwarning("No Subfolders", 
                                   "Target folder has no subfolders.\n\n"
                                   "Please create subfolders first.\n"
                                   "Example:\n"
                                   f"  {target}/Work/\n"
                                   f"  {target}/Personal/\n"
                                   f"  {target}/Archive/")
            return
        
        # Get all files
        files = []
        for file in os.listdir(source):
            file_path = os.path.join(source, file)
            if os.path.isfile(file_path):
                ext = Path(file_path).suffix.lower()
                if ext in ['.pdf', '.txt']:
                    files.append(file_path)
        
        if not files:
            messagebox.showinfo("No Files", "No PDF or TXT files found in source folder")
            return
        
        # Show confirmation
        folder_list = "\n".join([f"  • {f}" for f in subfolders])
        confirm = messagebox.askyesno(
            "Confirm Sorting",
            f"Source: {source}\n"
            f"Target: {target}\n\n"
            f"Found {len(files)} files to sort\n"
            f"Target subfolders:\n{folder_list}\n\n"
            f"Continue?"
        )
        
        if not confirm:
            return
        
        # Run sorting in thread
        self.run_sort_btn.config(state="disabled", text="⏳ Sorting in progress...")
        self.sort_log.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self._process_files, args=(files, subfolders, source, target), daemon=True)
        thread.start()
        
    def _process_files(self, files, subfolders, source, target):
        """Process files in batch with HITL support"""
        total = len(files)
        processed = 0
        auto_count = 0
        manual_count = 0
        failed_count = 0
        
        self._update_sort_log(f"🚀 Starting to process {total} files...\n")
        self._update_sort_log(f"📁 Target folders: {', '.join(subfolders)}\n")
        self._update_sort_log(f"🧠 Using KNN + NLI hybrid classification\n")
        self._update_sort_log(f"📄 Poppler path: {POPPLER_PATH}\n")
        self._update_sort_log("=" * 60 + "\n\n")
        
        for file_path in files:
            self._update_sort_log(f"\n[{processed + 1}/{total}] 📄 {os.path.basename(file_path)}...\n")
            
            # Process the file
            result = self.processor.process_file(file_path, subfolders)
            
            if result["success"]:
                if result["needs_hitl"]:
                    self._update_sort_log(f"  ⚠️ Low confidence ({result['confidence']:.2f}) - Manual input needed\n")
                    
                    selected_folder = self._show_hitl_dialog(file_path, subfolders, result.get("text", ""))
                    
                    if selected_folder and selected_folder in subfolders:
                        target_path = os.path.join(target, selected_folder)
                        self._move_file(file_path, target_path)
                        manual_count += 1
                        self._update_sort_log(f"  ✅ Manually sorted to: {selected_folder}\n")
                        self._add_activity(f"👤 {os.path.basename(file_path)} → {selected_folder} (Manual)")
                        
                        # Update KNN model
                        if INTEGRATED_AVAILABLE and result.get("text"):
                            try:
                                doc_embedding = embedder.encode(result["text"], convert_to_numpy=True)
                                update_model(selected_folder, doc_embedding)
                                self._update_sort_log(f"  🧠 KNN model updated with this decision\n")
                            except:
                                pass
                    else:
                        failed_count += 1
                        self._update_sort_log(f"  ❌ Skipped by user\n")
                else:
                    prediction = result["prediction"]
                    if prediction and prediction in subfolders:
                        target_path = os.path.join(target, prediction)
                        self._move_file(file_path, target_path)
                        auto_count += 1
                        self._update_sort_log(f"  ✅ Auto-sorted to: {prediction} (conf: {result['confidence']:.2f})\n")
                        self._add_activity(f"✓ {os.path.basename(file_path)} → {prediction} ({result['confidence']:.2f})")
                        
                        # Update KNN model
                        if INTEGRATED_AVAILABLE and result.get("text"):
                            try:
                                doc_embedding = embedder.encode(result["text"], convert_to_numpy=True)
                                update_model(prediction, doc_embedding)
                            except:
                                pass
                    else:
                        # Show HITL for invalid prediction
                        self._update_sort_log(f"  ⚠️ Prediction '{prediction}' not in target folders\n")
                        selected_folder = self._show_hitl_dialog(file_path, subfolders, result.get("text", ""))
                        
                        if selected_folder and selected_folder in subfolders:
                            target_path = os.path.join(target, selected_folder)
                            self._move_file(file_path, target_path)
                            manual_count += 1
                            self._update_sort_log(f"  ✅ Manually sorted to: {selected_folder}\n")
                            self._add_activity(f"👤 {os.path.basename(file_path)} → {selected_folder} (Manual)")
                        else:
                            failed_count += 1
                            self._update_sort_log(f"  ❌ Skipped\n")
            else:
                failed_count += 1
                self._update_sort_log(f"  ❌ Error: {result.get('error', 'Unknown')}\n")
            
            processed += 1
            progress = (processed / total) * 100
            self._update_progress(progress, processed, total, auto_count, manual_count, failed_count)
        
        # Update KNN memory stat
        self._update_knn_memory_stat()
        
        # Complete
        self._update_sort_log(f"\n{'='*60}\n")
        self._update_sort_log(f"✅ SORTING COMPLETE!\n")
        self._update_sort_log(f"  Auto-sorted (KNN+NLI): {auto_count}\n")
        self._update_sort_log(f"  Manual (HITL): {manual_count}\n")
        self._update_sort_log(f"  Failed: {failed_count}\n")
        self._update_sort_log(f"  Total processed: {total}\n")
        
        self._update_stats_after_sort(auto_count, manual_count, failed_count)
        
        self.after(0, lambda: self.run_sort_btn.config(state="normal", text="▶ RUN SORTING"))
        self.after(0, lambda: messagebox.showinfo("Complete", 
                                                   f"Sorting complete!\n\n"
                                                   f"✓ Auto-sorted (KNN+NLI): {auto_count}\n"
                                                   f"👤 Manual (HITL): {manual_count}\n"
                                                   f"❌ Failed: {failed_count}\n"
                                                   f"📊 Total: {total}"))
        
    def _show_hitl_dialog(self, file_path, folders, text_content):
        """Show HITL dialog and wait for result"""
        import queue
        result_queue = queue.Queue()
        
        def show_dialog():
            dialog = HITLDialog(self, os.path.basename(file_path), folders, file_path, text_content)
            result_queue.put(dialog.selected_folder)
        
        self.after(0, show_dialog)
        return result_queue.get()
        
    def _move_file(self, source_path, target_folder):
        """Move file to target folder"""
        try:
            filename = os.path.basename(source_path)
            dest_path = os.path.join(target_folder, filename)
            
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                timestamp = int(time.time())
                dest_path = os.path.join(target_folder, f"{base}_{timestamp}{ext}")
            
            shutil.move(source_path, dest_path)
            return True
        except Exception as e:
            print(f"Move error: {e}")
            return False
            
    def _add_activity(self, message):
        """Add activity to dashboard list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.after(0, lambda: self.activity_list.insert(0, f"  [{timestamp}] {message}"))
        self.after(0, lambda: self.activity_list.delete(100, tk.END))
        
    def _update_knn_memory_stat(self):
        """Update KNN memory stat"""
        try:
            from integrated import cached_labels
            memory_size = len(cached_labels) if cached_labels else 0
            self.after(0, lambda: self.stat_labels["KNN Memory"].config(text=str(memory_size)))
        except:
            pass
        
    def _update_stats(self):
        """Update dashboard statistics"""
        items = self.activity_list.get(0, tk.END)
        total = len([i for i in items if "→" in i])
        auto = len([i for i in items if "✓" in i and "Manual" not in i])
        manual = len([i for i in items if "Manual" in i])
        failed = len([i for i in items if "❌" in i])
        
        self.stat_labels["Files Sorted"].config(text=str(total))
        self.stat_labels["Auto Sorted"].config(text=str(auto))
        self.stat_labels["Manual (HITL)"].config(text=str(manual))
        self.stat_labels["Failed"].config(text=str(failed))
        self._update_knn_memory_stat()
        
    def _update_stats_after_sort(self, auto, manual, failed):
        """Update stats after batch sort"""
        current_total = int(self.stat_labels["Files Sorted"].cget("text"))
        current_auto = int(self.stat_labels["Auto Sorted"].cget("text"))
        current_manual = int(self.stat_labels["Manual (HITL)"].cget("text"))
        current_failed = int(self.stat_labels["Failed"].cget("text"))
        
        self.stat_labels["Files Sorted"].config(text=str(current_total + auto + manual))
        self.stat_labels["Auto Sorted"].config(text=str(current_auto + auto))
        self.stat_labels["Manual (HITL)"].config(text=str(current_manual + manual))
        self.stat_labels["Failed"].config(text=str(current_failed + failed))
        self._update_knn_memory_stat()
        
    def _update_sort_log(self, message):
        """Update sort log from background thread"""
        self.after(0, lambda: self.sort_log.insert(tk.END, message))
        self.after(0, lambda: self.sort_log.see(tk.END))
        
    def _update_progress(self, progress, processed, total, auto, manual, failed):
        """Update progress bar and status"""
        def update():
            self.progress_var.set(progress)
            self.sort_status.config(text=f"Progress: {processed}/{total} | ✓ KNN+NLI: {auto} | 👤 Manual: {manual} | ❌ Failed: {failed}")
        
        self.after(0, update)
        
    def _update_status(self, message):
        """Update status indicator"""
        self.status_text.config(text=f"  {message}")

# =========================
# MAIN
# =========================

def main():
    """Main entry point"""
    print("=" * 50)
    print("NLI File Sorter - Intelligent Document Organizer")
    print("Using KNN + NLI Hybrid Classification")
    print("=" * 50)
    print(f"Poppler path: {POPPLER_PATH}")
    print(f"Poppler available: {POPPLER_AVAILABLE}")
    print(f"Tesseract available: {TESSERACT_AVAILABLE}")
    print("=" * 50)
    
    if not INTEGRATED_AVAILABLE:
        print("⚠️ Warning: integrated.py not available.")
        print("Make sure integrated.py is in the same directory.")
    
    if not OCR_AVAILABLE:
        print("⚠️ Warning: OCR libraries not fully installed.")
        print("Install with: pip install pytesseract pdf2image pillow")
    
    if POPPLER_AVAILABLE:
        print("✓ Poppler is configured and ready for OCR!")
    else:
        print(f"⚠️ Poppler not found at: {POPPLER_PATH}")
        print("   Please verify the path is correct")
    
    print("\nStarting application...")
    app = NLIApplication()
    app.mainloop()

if __name__ == "__main__":
    main()