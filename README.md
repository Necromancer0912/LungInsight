<div align="center">

# 🫁 LungInsight

### AI-Powered Lung Disease Detection & Clinical Decision Support System

An intelligent diagnostic platform combining deep learning-based audio and image analysis with LLM-driven clinical reasoning for comprehensive respiratory disease assessment.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-17.0+-61DAFB.svg)](https://reactjs.org/)
[![Electron](https://img.shields.io/badge/Electron-Desktop-47848F.svg)](https://www.electronjs.org/)

[Features](#-features) • [Demo](#-demo) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Contact](#-contact)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Machine Learning Models](#-machine-learning-models)
- [Configuration](#-configuration)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

LungInsight is a comprehensive diagnostic platform that bridges the gap between AI-powered disease detection and clinical decision-making. The system analyzes respiratory audio recordings and chest X-ray images to identify potential lung conditions, then leverages large language models to generate detailed clinical assessments and actionable recommendations.

### What We're Doing

**The Problem:** Traditional diagnostic workflows often miss critical follow-up questions, provide vague recommendations, and lack structured clinical reasoning that connects AI predictions to actionable medical insights.

**Our Solution:** LungInsight integrates:
- **Multi-Modal Analysis**: Process both lung sounds (audio) and chest X-rays (images)
- **Deep Learning Detection**: Identify conditions like COPD, pneumonia, asthma, bronchitis, and more
- **Intelligent Follow-Up**: Generate 16+ detailed, condition-specific clinical questions
- **Structured Reporting**: Create comprehensive medical reports with severity assessment, red flags, treatment recommendations, and escalation criteria
- **Privacy-First Design**: All processing happens locally—no data leaves your machine

### Key Capabilities

✅ **Dual-Modal Detection**: Audio breath analysis + Chest X-ray classification  
✅ **LLM-Enhanced Reasoning**: Context-aware clinical question generation  
✅ **Structured Reports**: Severity scoring, medication guidance with contraindications  
✅ **Local Processing**: Privacy-preserving, offline-capable architecture  
✅ **Clinical Focus**: Red flag identification, differential diagnosis, escalation triggers  

---

## ✨ Features

### 🔬 Advanced Diagnostics
- **Audio Analysis**: CNN-based breath sound classification for respiratory conditions
- **Image Analysis**: EfficientNet-B4/B7 architecture for chest X-ray interpretation
- **Multi-Class Detection**: COPD, Pneumonia, Asthma, Bronchitis, URTI, Bronchiolitis, and Healthy classifications

### 🤖 AI-Powered Clinical Reasoning
- **Smart Questionnaire**: Automatically generates 16 detailed, disease-specific follow-up questions
- **Comprehensive Reports**: Structured analysis including:
  - Clinical summary and likely conditions
  - Severity assessment (Low/Moderate/High)
  - Red flag identification
  - Recommended diagnostic tests with rationale
  - Medication guidance (dosing, contraindications, interactions)
  - Lifestyle and dietary recommendations
  - Escalation triggers for emergency care
  - Differential diagnosis considerations

### 🎨 Modern User Experience
- **Desktop Application**: Native Electron-based interface for Windows
- **Intuitive Workflow**: Three-step process (Intake → Follow-up → Report)
- **Real-time Feedback**: Progress indicators and immediate analysis
- **Clean Design**: Modern typography (Space Grotesk), gradient accents, structured layouts

### 🔒 Privacy & Security
- **Local-First Architecture**: All data processing happens on your machine
- **No External APIs**: Uses local Ollama for LLM operations
- **Secure Storage**: Uploads stored locally and excluded from version control
- **HIPAA-Conscious Design**: No PHI transmission or cloud storage

---

## 🎬 Demo

### Application Workflow

```
┌─────────────────┐
│  Step 1: Intake │  Upload audio/image + patient demographics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AI Analysis     │  Deep learning model processes input
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Step 2: Q&A     │  16 intelligent follow-up questions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Step 3: Report  │  Comprehensive clinical assessment
└─────────────────┘
```

### Sample Output

**Input**: Chest X-ray showing infiltrates  
**Detection**: Pneumonia (Confidence: 94%)  
**Generated Questions**: 16 detailed questions about onset, fever patterns, comorbidities, current medications, etc.  
**Final Report**:
- Severity: Moderate  
- Red Flags: Fever >102°F for 3 days, productive cough  
- Recommended Tests: Complete blood count, CRP, sputum culture  
- Medications: Amoxicillin-clavulanate 875mg BID (avoid if penicillin allergy)  
- Escalation Triggers: Persistent fever >5 days, worsening dyspnea  

### Video Demo

> 📹 A complete walkthrough video (`demo.mp4`) is included in the repository, demonstrating:
> - Patient intake and file upload
> - Real-time disease detection
> - Interactive clinical questionnaire
> - Comprehensive report generation with severity assessment

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: React 17 with Hooks
- **Desktop Shell**: Electron (multi-platform desktop application)
- **Styling**: styled-components for CSS-in-JS
- **Build Tools**: Webpack 5, Babel 7
- **HTTP Client**: Axios for API communication
- **Testing**: Jest + React Testing Library

### Backend
- **Framework**: Flask (Python web framework)
- **CORS**: Flask-CORS for cross-origin requests
- **Machine Learning**: PyTorch, scikit-learn, librosa
- **Image Processing**: OpenCV, PIL
- **Audio Processing**: librosa, soundfile
- **LLM Integration**: Ollama client (local inference)

### Machine Learning
- **Audio Model**: Custom CNN architecture for breath sound classification
- **Image Model**: EfficientNet (B4/B7) transfer learning
- **LLM**: Qwen 2.5 (3B/7B parameters) via Ollama
- **Frameworks**: PyTorch, TensorFlow/Keras

### Development Tools
- **Version Control**: Git
- **Package Management**: npm (frontend), pip (backend)
- **Environment Management**: Python venv
- **Code Quality**: ESLint, Prettier

---

## 🏗️ Architecture

### System Design

```
┌──────────────────────────────────────────────────────────┐
│                    Electron Desktop App                   │
│  ┌────────────────────────────────────────────────────┐  │
│  │           React UI (Renderer Process)              │  │
│  │  • File Upload  • Patient Intake  • Q&A Interface │  │
│  │  • Report Parsing  • Severity Logic  • Styling    │  │
│  └────────────────┬───────────────────────────────────┘  │
│                   │ IPC (Secure Preload Bridge)          │
│  ┌────────────────▼───────────────────────────────────┐  │
│  │           Electron Main Process                    │  │
│  │  • Window Management  • File System Access         │  │
│  └────────────────────────────────────────────────────┘  │
└───────────────────┬──────────────────────────────────────┘
                    │ HTTP/REST API
┌───────────────────▼──────────────────────────────────────┐
│                  Flask Backend Server                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  API Routes                                        │  │
│  │  • /health  • /audio_prediction                    │  │
│  │  • /image_prediction  • /generate_questions        │  │
│  │  • /analyze_responses                              │  │
│  └───────┬─────────────────────────┬──────────────────┘  │
│          │                         │                      │
│  ┌───────▼──────────┐     ┌────────▼─────────────────┐  │
│  │ ML Models        │     │ LLM Integration          │  │
│  │ • Audio CNN      │     │ • Ollama Client          │  │
│  │ • Image EfficNet │     │ • Prompt Engineering     │  │
│  └──────────────────┘     └──────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Intake Phase**: User uploads audio/image file via Electron UI
2. **Transmission**: File sent to Flask backend via multipart/form-data
3. **Prediction**: Appropriate ML model processes input and returns diagnosis
4. **Question Generation**: LLM generates targeted follow-up questions based on diagnosis
5. **User Response**: Patient/clinician answers questions in UI
6. **Report Generation**: LLM synthesizes responses into structured clinical report
7. **Rendering**: Frontend parses markdown report into formatted sections with severity badges

---

## 📁 Project Structure

```
LungInsight/
│
├── electron_app/               # Desktop application (Electron + React)
│   ├── src/
│   │   ├── App.js             # Main React component with UI logic
│   │   ├── App.test.js        # Jest test suite
│   │   └── index.js           # React entry point
│   ├── test/
│   │   ├── setupTests.js      # Testing configuration
│   │   └── styleMock.js       # Style import mocks
│   ├── index.html             # HTML shell
│   ├── main.js                # Electron main process
│   ├── preload.js             # Secure IPC bridge
│   ├── webpack.config.js      # Build configuration
│   ├── tailwind.config.js     # Tailwind CSS config
│   ├── postcss.config.js      # PostCSS configuration
│   ├── jest.config.js         # Jest configuration
│   ├── .babelrc               # Babel transpiler config
│   ├── package.json           # Node dependencies
│   └── package-lock.json      # Locked dependency versions
│
├── flask_server/              # Backend API server
│   ├── app.py                 # Flask routes and LLM orchestration
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example           # Environment variable template
│   └── uploaded_files/        # Runtime file storage (git-ignored)
│
├── detection_model/           # Machine learning models
│   ├── Audio_model.py         # Audio classification pipeline
│   ├── Image_model.py         # Image classification pipeline
│   ├── Image/                 # Image model development
│   │   ├── Image_model.py
│   │   ├── Image_model.ipynb
│   │   ├── image_classification.ipynb
│   │   ├── ads.py
│   │   └── model.pth          # Pre-trained weights (git-ignored)
│   ├── Models/                # Trained model checkpoints
│   │   ├── model.pth          # Latest model (git-ignored)
│   │   └── classification_model_train_49_82.5_test_91.6.pth
│   └── Architecture/          # Model architecture diagrams
│       ├── audio_model.gv
│       ├── audio_model.gv.pdf
│       ├── audio_model.gv.png
│       └── audio_model.gv.svg
│
├── archive/                   # Legacy code and experiments
│   ├── legacy_backend/
│   ├── legacy_electron/
│   ├── legacy_models/
│   └── legacy_scratch/
│
├── .gitignore                 # Git exclusion rules
├── README.md                  # This file
└── demo.mp4                   # Application demo video
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `electron_app/` | Frontend desktop application built with Electron and React |
| `flask_server/` | Backend REST API server handling ML inference and LLM operations |
| `detection_model/` | Machine learning model implementations and weights |
| `archive/` | Historical code and experimental features (not in production) |

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 14.x or higher
- **npm**: 6.x or higher
- **Ollama**: For local LLM inference ([Download](https://ollama.ai))
- **Git**: For version control

### Step 1: Clone Repository

```bash
git clone https://github.com/Necromancer0912/LungInsight.git
cd LungInsight
```

### Step 2: Backend Setup

```powershell
# Navigate to Flask server directory
cd flask_server

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

### Step 3: Install Ollama & Download Model

```powershell
# Install Ollama from https://ollama.ai

# Pull the required LLM model
ollama pull qwen2.5:3b

# Optional: Use larger model for better results
ollama pull qwen2.5:7b
```

### Step 4: Frontend Setup

```powershell
# Navigate to Electron app directory
cd ../electron_app

# Install Node dependencies
npm install

# Build the application
npm run build
```

### Step 5: Model Weights

Ensure your trained model weights are placed in the appropriate directories:
- Audio model: `detection_model/Models/`
- Image model: `detection_model/Image/model.pth`

> **Note**: Model weights are not included in the repository due to file size. Train your models using the provided notebooks or contact the author for pre-trained weights.

---

## 💻 Usage

### Starting the Application

#### Terminal 1: Start Flask Backend

```powershell
cd flask_server
.\venv\Scripts\Activate.ps1
python app.py
```

The backend will start on `http://localhost:5000`

#### Terminal 2: Start Electron App

```powershell
cd electron_app
npm start
```

The desktop application will launch automatically.

### Using LungInsight

1. **Patient Intake**
   - Enter patient demographics (name, age, gender)
   - Select modality: Audio (breath sounds) or Image (chest X-ray)
   - Upload the relevant file

2. **AI Analysis**
   - System processes the file and returns a diagnosis
   - View detected condition and confidence level

3. **Clinical Questionnaire**
   - Answer 16 intelligent follow-up questions
   - Questions are tailored to the detected condition
   - Progress bar shows completion status
   - Option to skip questions if needed

4. **Comprehensive Report**
   - Review structured clinical assessment
   - Sections include: Summary, Severity, Red Flags, Tests, Medications, Diet, Escalation Triggers
   - Severity badge (Low/Moderate/High) with color coding
   - Export or print report for medical records

---

## 📡 API Reference

### Base URL

```
http://localhost:5000
```

### Endpoints

#### Health Check

```http
GET /health
```

**Response**
```json
{
  "status": "ok"
}
```

#### Audio Prediction

```http
POST /audio_prediction
```

**Request**
- Content-Type: `multipart/form-data`
- Body: `file` (audio file)

**Response**
```json
{
  "prediction": "COPD"
}
```

#### Image Prediction

```http
POST /image_prediction
```

**Request**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response**
```json
{
  "prediction": "Pneumonia"
}
```

#### Generate Questions

```http
POST /generate_questions
```

**Request**
```json
{
  "disease": "Pneumonia"
}
```

**Response**
```json
{
  "questions": [
    "1. When did you first notice symptoms?",
    "2. Have you experienced fever? If so, how high?",
    "...",
    "16. Are you currently taking any medications?"
  ],
  "warning": "Optional warning message"
}
```

#### Analyze Responses

```http
POST /analyze_responses
```

**Request**
```json
{
  "answers": {
    "0": "Symptoms started 3 days ago",
    "1": "Yes, fever of 102°F",
    "..."
  }
}
```

**Response**
```json
{
  "analysis": "**Summary:**\n\nPatient presents with...\n\n**Severity:** Moderate\n\n..."
}
```

---

## 🧠 Machine Learning Models

### Audio Classification Model

**Architecture**: Custom Convolutional Neural Network (CNN)

**Features**:
- Input: Audio waveform (breath sounds)
- Preprocessing: Mel-spectrogram transformation using librosa
- Layers: Multiple convolutional blocks with batch normalization and dropout
- Output: Multi-class probability distribution

**Classes**: COPD, Asthma, Bronchitis, URTI, Bronchiolitis, Healthy

**Performance**:
- Training Accuracy: 82.5%
- Test Accuracy: 91.6%

### Image Classification Model

**Architecture**: EfficientNet-B4/B7 (Transfer Learning)

**Features**:
- Input: Chest X-ray images (224x224 or 299x299)
- Preprocessing: Normalization, resizing
- Transfer Learning: Pre-trained on ImageNet, fine-tuned on medical dataset
- Output: Disease classification with confidence scores

**Classes**: Pneumonia, COPD, Bronchitis, Healthy, and others

**Performance**: Optimized for medical imaging with data augmentation and class balancing

### LLM Integration

**Model**: Qwen 2.5 (3B or 7B parameters)

**Tasks**:
1. **Question Generation**: Creates 16 targeted clinical questions based on diagnosed condition
2. **Report Synthesis**: Analyzes patient responses and generates structured medical reports

**Prompt Engineering**:
- Specialized medical prompts for clinical accuracy
- Structured output formatting for consistent parsing
- Context-aware reasoning for differential diagnosis

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in `flask_server/`:

```env
# LLM Configuration
OLLAMA_MODEL=qwen2.5:3b          # Options: qwen2.5:3b, qwen2.5:7b

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# API Configuration
REACT_APP_API_BASE=http://localhost:5000
```

### Frontend Configuration

Edit `electron_app/webpack.config.js` to change API endpoint:

```javascript
new webpack.DefinePlugin({
  'process.env.REACT_APP_API_BASE': JSON.stringify('http://localhost:5000')
})
```

### Model Configuration

Adjust model paths in respective model files:
- Audio: `detection_model/Audio_model.py`
- Image: `detection_model/Image_model.py`

---

## 🔧 Development

### Running Tests

#### Frontend Tests

```powershell
cd electron_app
npm test
```

#### Manual Testing Workflow

1. Start backend: `python flask_server/app.py`
2. Start frontend: `npm start` in `electron_app/`
3. Test each workflow step:
   - File upload (audio and image)
   - Prediction accuracy
   - Question generation
   - Report parsing and formatting

### Building for Production

```powershell
cd electron_app
npm run build
```

Output will be in `electron_app/dist/`

### Code Style

- **Frontend**: ESLint + Prettier (React/JavaScript)
- **Backend**: PEP 8 (Python)
- **Formatting**: Use provided config files

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement changes with tests
3. Run linting and tests
4. Submit pull request

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Blank Electron window | JavaScript error in renderer | Check browser console with `ELECTRON_ENABLE_LOGGING=1` |
| 500 error on predictions | Missing model weights | Ensure model files exist in `detection_model/Models/` |
| Empty LLM responses | Ollama not running | Start Ollama service, verify model is downloaded |
| API unreachable | Flask not started | Start Flask server: `python app.py` |
| Module import errors | Dependencies not installed | Run `pip install -r requirements.txt` |
| Build failures | Node modules issue | Delete `node_modules/`, run `npm install` again |

### Debug Mode

Enable verbose logging:

```powershell
# Electron
$env:ELECTRON_ENABLE_LOGGING=1
npm start

# Flask
$env:FLASK_DEBUG=True
python app.py
```

### Support

If issues persist:
1. Check console/terminal output for error messages
2. Verify all dependencies are correctly installed
3. Ensure model files are in correct locations
4. Contact project maintainer (see [Contact](#-contact))

---

## 🗺️ Roadmap

### Current Version: 1.0.0

### Planned Features

- [ ] **Multi-Language Support**: Internationalization for global use
- [ ] **Cloud Deployment**: Optional cloud-based inference for resource-constrained devices
- [ ] **Mobile App**: iOS/Android companion app
- [ ] **Report Export**: PDF/DOCX export functionality
- [ ] **Patient History**: Session persistence and historical tracking
- [ ] **Integration**: FHIR/HL7 compatibility for EHR systems
- [ ] **Enhanced Models**: Continual learning and model updates
- [ ] **Telemetry**: Optional anonymous usage analytics for model improvement
- [ ] **Packaging**: Electron Builder for distributable installers
- [ ] **CI/CD**: Automated testing and deployment pipeline

### Research Directions

- Multi-modal fusion (combining audio + image for single patient)
- Explainable AI visualizations
- Real-time audio monitoring
- Edge deployment for IoT devices

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Areas

- 🐛 Bug fixes and issue resolution
- ✨ New feature implementation
- 📝 Documentation improvements
- 🧪 Test coverage expansion
- 🎨 UI/UX enhancements
- 🔬 Model accuracy improvements

### Code Standards

- Follow existing code style
- Write meaningful commit messages
- Include tests for new features
- Update documentation accordingly
- Ensure all tests pass before submitting PR

### Reporting Issues

Use GitHub Issues to report bugs or request features. Please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, etc.)

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Sayan Das

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact

**Sayan Das**

- 📧 Email: [sayan20012002@gmail.com](mailto:sayan20012002@gmail.com)
- 💼 GitHub: [@Necromancer0912](https://github.com/Necromancer0912)
- 🔗 Project Link: [https://github.com/Necromancer0912/LungInsight](https://github.com/Necromancer0912/LungInsight)

### Acknowledgments

- EfficientNet architecture by Google Research
- Qwen LLM by Alibaba Cloud
- Ollama for local LLM inference
- Electron and React communities
- Medical imaging datasets and respiratory audio databases

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star!

Made with ❤️ by Sayan Das

</div>
