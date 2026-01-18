export interface PatientData {
  // Patient Details
  patientId?: string;
  name?: string;
  age: number;
  gender: 'Male' | 'Female' | '';

  // Clinical Parameters (ILPD Dataset)
  totalBilirubin: number;
  directBilirubin: number;
  alkalinePhosphatase: number;
  sgptAlt: number;
  sgotAst: number;
  totalProteins: number;
  albumin: number;
  agRatio: number;
}

export interface PredictionResult {
  hasDiseaseRisk: boolean;
  probability: number;
  confidence: 'Low' | 'Medium' | 'High';
  contributingFactors: ContributingFactor[];
  clinicalInterpretation: string;
  recommendations: string[];
  timestamp: Date;
}

export interface ContributingFactor {
  feature: string;
  patientValue: number;
  normalRange: string;
  contribution: number; // 0-1
  contributionLevel: 'High' | 'Medium' | 'Low';
  status: 'Normal' | 'High' | 'Low';
  clinicalNote: string;
}

export interface NormalRange {
  min: number;
  max: number;
  unit: string;
  description: string;
}

export const NORMAL_RANGES: Record<string, NormalRange> = {
  totalBilirubin: { min: 0.1, max: 1.2, unit: 'mg/dL', description: 'Normal range: 0.1 – 1.2 mg/dL' },
  directBilirubin: { min: 0.0, max: 0.3, unit: 'mg/dL', description: 'Normal range: 0.0 – 0.3 mg/dL' },
  alkalinePhosphatase: { min: 40, max: 130, unit: 'IU/L', description: 'Normal range: 40 – 130 IU/L' },
  sgptAlt: { min: 7, max: 56, unit: 'IU/L', description: 'Normal range: 7 – 56 IU/L' },
  sgotAst: { min: 10, max: 40, unit: 'IU/L', description: 'Normal range: 10 – 40 IU/L' },
  totalProteins: { min: 6.0, max: 8.3, unit: 'g/dL', description: 'Normal range: 6.0 – 8.3 g/dL' },
  albumin: { min: 3.5, max: 5.5, unit: 'g/dL', description: 'Normal range: 3.5 – 5.5 g/dL' },
  agRatio: { min: 1.0, max: 2.5, unit: '', description: 'Normal range: 1.0 – 2.5' },
};
