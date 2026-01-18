# Liver Disease Prediction System - ILPD ML Dashboard

## Overview
This is a professional, doctor-friendly web application for liver disease risk prediction using machine learning. The system is designed as a clinical decision support tool for healthcare professionals and laboratory staff.

## ⚠️ Medical Disclaimer
**This tool provides ML-based risk prediction support. It is NOT a medical diagnosis. Final diagnosis and treatment decisions must be made by qualified healthcare professionals.**

## Features

### 1. Patient Input Form
- **Patient Demographics**: Optional fields for Patient ID, Name, Age, and Gender
- **Clinical Laboratory Parameters** (ILPD Dataset):
  - Total Bilirubin (TB) - mg/dL
  - Direct Bilirubin (DB) - mg/dL
  - Alkaline Phosphatase (Alkphos) - IU/L
  - SGPT / ALT (Alamine Aminotransferase) - IU/L
  - SGOT / AST (Aspartate Aminotransferase) - IU/L
  - Total Proteins (TP) - g/dL
  - Albumin (ALB) - g/dL
  - Albumin/Globulin Ratio (A/G Ratio)

- **Smart Features**:
  - Real-time input validation
  - Normal range tooltips on hover
  - Visual highlighting of abnormal values (amber background)
  - "Fill Example Values" buttons (Disease/Healthy)
  - Reset functionality

### 2. Results Dashboard

#### Card A: Prediction Summary
- Risk Status Badge (Disease/No Disease)
- Risk Probability Percentage with visual progress bar
- Confidence Level (High/Medium/Low)
- Color-coded display (Red for risk, Green for healthy)

#### Card B: Key Contributing Factors (Model Explainability)
- Shows top 5 factors when disease risk is detected
- For each factor displays:
  - Feature name and patient value
  - Normal range reference
  - Status indicator (High/Low/Normal) with arrows
  - Contribution strength bar (High/Medium/Low impact)
  - Clinical interpretation note
- Based on SHAP-like explainability methodology

#### Card C: Clinical Interpretation
- Comprehensive analysis of laboratory findings
- Pattern-based clinical insights
- Phrased as "may indicate" and "possible" (not definitive)
- Considers combinations of abnormal parameters

#### Card D: Visual Charts
- **Bar Chart**: Patient values vs. normal upper limits
- **Pie Chart**: Contribution weights of each factor
- Interactive tooltips with detailed information
- Tabbed interface for easy navigation

#### Card E: Clinical Recommendations
- Checklist-style recommendations for physician review
- Dynamic recommendations based on:
  - Abnormal parameter patterns
  - Risk probability level
  - Specific clinical concerns
- Includes follow-up actions like:
  - Repeat LFT panel
  - Hepatitis screening
  - Imaging studies
  - Specialist referrals
- Export options: Copy Summary, Export PDF, Save to Record

### 3. Navigation & Interface

#### Header
- Application title and subtitle
- Real-time date and time display
- Clinical System badge
- Navigation: Home | Predict | About Model | Help
- **Important Disclaimer Modal** - Comprehensive medical disclaimer

#### About Model Section
- Dataset Information (ILPD)
- Machine Learning Model Details
- Explainability Methods (SHAP/LIME)
- Clinical Usage Guidelines
- Important Limitations
- Technical Specifications

## Technical Architecture

### Frontend Stack
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: Radix UI primitives
- **Charts**: Recharts library
- **Icons**: Lucide React
- **Animations**: Motion (Framer Motion)
- **Notifications**: Sonner (toast notifications)

### Component Structure
```
/src/app/
├── App.tsx                          # Main application component
├── components/
│   ├── DashboardHeader.tsx          # Top navigation with disclaimer
│   ├── PatientInputForm.tsx         # Left panel input form
│   ├── ResultsDashboard.tsx         # Right panel results container
│   ├── PredictionSummary.tsx        # Card A - Prediction result
│   ├── ContributingFactors.tsx      # Card B - Explainability
│   ├── ClinicalInterpretation.tsx   # Card C - Clinical analysis
│   ├── VisualCharts.tsx             # Card D - Charts
│   ├── Recommendations.tsx          # Card E - Clinical recommendations
│   └── AboutSection.tsx             # About page content
├── types/
│   └── patient.ts                   # TypeScript interfaces and constants
└── utils/
    └── mlModel.ts                   # ML prediction logic (mock)
```

### Key Files

#### `/src/app/types/patient.ts`
Defines all TypeScript interfaces:
- `PatientData` - Input data structure
- `PredictionResult` - ML output structure
- `ContributingFactor` - Explainability data
- `NORMAL_RANGES` - Reference ranges for all parameters

#### `/src/app/utils/mlModel.ts`
Mock ML prediction engine with:
- `predictLiverDisease()` - Main prediction function
- `getExamplePatientData()` - Sample disease case
- `getHealthyPatientData()` - Sample healthy case
- Risk scoring algorithm based on parameter deviations
- Clinical interpretation generation
- Recommendation generation

## Design System

### Color Scheme
- **Primary**: Teal/Cyan gradient (medical professional theme)
- **Success**: Green (#22c55e)
- **Warning**: Amber (#f59e0b)
- **Danger**: Red (#dc2626)
- **Background**: Soft gray-blue gradient

### Medical UI Principles
- Clean, minimal, no clutter
- Ample white space
- Card-based sections
- Professional color palette
- Clear visual hierarchy
- Accessibility-first design

### Responsive Design
- Desktop: Two-column layout (Form | Results)
- Tablet: Optimized single column with sticky form
- Mobile: Stacked layout with smooth scrolling

## Machine Learning Model (Mock Implementation)

### Current Implementation
The current version uses a **mock ML model** that simulates:
1. Risk scoring based on deviation from normal ranges
2. Weighted contribution of each parameter
3. SHAP-like explainability
4. Clinical interpretation generation

### Production Integration Points
For real ML integration, replace `/src/app/utils/mlModel.ts` with:
- API endpoint calls to your ML service
- Authentication/authorization
- Error handling
- Loading states
- Real SHAP/LIME explainability from your model

### Recommended ML Models
- Random Forest Classifier
- XGBoost
- Logistic Regression
- Neural Networks (with explainability wrapper)

## Usage Guidelines

### For Healthcare Providers
✅ **Appropriate Uses:**
- Screening and risk stratification
- Supplement to clinical evaluation
- Identifying patients needing further workup
- Educational tool for medical trainees

❌ **Not Suitable For:**
- Standalone diagnosis without clinical correlation
- Emergency or critical care decisions
- Replacing comprehensive patient evaluation
- Legal or regulatory documentation

### Data Privacy
- Current demo: No data storage or transmission
- Production: Must implement HIPAA-compliant data handling
- Consider on-premise deployment for sensitive environments

## Installation & Setup

### Prerequisites
- Node.js 18+ 
- npm or pnpm

### Install Dependencies
```bash
npm install
# or
pnpm install
```

### Development
```bash
npm run dev
```

### Build for Production
```bash
npm run build
```

## Customization

### Adjusting Normal Ranges
Edit `/src/app/types/patient.ts` - `NORMAL_RANGES` object

### Modifying ML Logic
Edit `/src/app/utils/mlModel.ts` - `predictLiverDisease()` function

### Styling Changes
- Global styles: `/src/styles/theme.css`
- Tailwind config: Uses Tailwind v4 (no config file needed)
- Component-specific: Inline Tailwind classes

### Adding New Parameters
1. Update `PatientData` interface in `types/patient.ts`
2. Add to `NORMAL_RANGES`
3. Add input field in `PatientInputForm.tsx`
4. Update ML logic in `mlModel.ts`
5. Update chart data in `VisualCharts.tsx`

## Accessibility Features
- Keyboard navigation support
- Proper ARIA labels
- Screen reader friendly
- High contrast ratios
- Focus indicators
- Semantic HTML structure

## Browser Support
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Known Limitations
1. Mock ML model (not trained on real data)
2. No backend persistence
3. No user authentication
4. No multi-language support
5. No print-optimized layout (PDF export is UI only)

## Future Enhancements
- [ ] Real ML model integration via API
- [ ] User authentication and authorization
- [ ] Patient record management system
- [ ] Historical predictions tracking
- [ ] Multi-language support
- [ ] Print/PDF generation with proper formatting
- [ ] Batch prediction for multiple patients
- [ ] Integration with hospital EMR systems
- [ ] Advanced analytics and reporting
- [ ] Mobile app version

## Security Considerations
- Validate all user inputs
- Implement rate limiting for API calls
- Use HTTPS in production
- Regular security audits
- Compliance with healthcare data regulations (HIPAA, GDPR)
- Secure ML model serving

## Support & Contact
For technical support, bug reports, or feature requests, please contact your system administrator.

## License
For demonstration and educational purposes. Not for clinical use without proper validation and regulatory approval.

---

**Version**: 1.0.0  
**Last Updated**: January 17, 2026  
**Developed for**: Clinical Decision Support & Medical AI Education
