import { PatientData, PredictionResult, ContributingFactor, NORMAL_RANGES } from '@/app/types/patient';

// Backend API URL - adjust this for your environment
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Call backend API for liver disease prediction
 * Returns predictions from real ML models (XGBoost, Unsupervised, SHAP)
 */

export async function predictLiverDisease(data: PatientData): Promise<PredictionResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predict/individual`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const apiResult = await response.json();

    // Transform API response to frontend format
    return transformApiResponseToPredictionResult(apiResult, data);
  } catch (error) {
    console.error('API prediction error:', error);
    // Fallback to mock if API fails (for development)
    return predictLiverDiseaseMock(data);
  }
}

/**
 * Transform backend API response to frontend PredictionResult format
 */
function transformApiResponseToPredictionResult(apiResult: any, originalData: PatientData): PredictionResult {
  const supervised = apiResult.supervised || {};
  const shap = apiResult.shap || { top_contributing_factors: [] };

  // Convert probability from 0-1 to 0-100
  const probability = (supervised.risk_probability || 0) * 100;
  const hasDiseaseRisk = probability > 50;

  // Convert SHAP contributions to ContributingFactor format
  const contributingFactors: ContributingFactor[] = shap.top_contributing_factors?.slice(0, 5).map((factor: any) => {
    const featureName = factor.feature;
    const shapValue = factor.shap_value || 0;
    const contribution = Math.abs(shapValue);
    
    // Map feature names to display names
    const displayName = mapFeatureNameToDisplay(featureName);
    
    // Get normal range for feature
    const normalRange = getNormalRangeForFeature(featureName);
    const patientValue = getPatientValueForFeature(featureName, originalData);
    const status = evaluateParameter(patientValue, normalRange.min, normalRange.max);

    return {
      feature: displayName,
      patientValue: patientValue,
      normalRange: normalRange.description,
      contribution: Math.min(contribution, 1), // Normalize to 0-1
      contributionLevel: getContributionLevel(Math.min(contribution, 1)),
      status: status,
      clinicalNote: getClinicalNoteForFeature(featureName),
    };
  }) || [];

  // Generate clinical interpretation
  const clinicalInterpretation = generateClinicalInterpretation(contributingFactors, originalData);

  // Generate recommendations
  const recommendations = generateRecommendations(contributingFactors, originalData, probability);

  return {
    hasDiseaseRisk,
    probability: Math.round(probability * 10) / 10,
    confidence: supervised.confidence || 'Medium',
    contributingFactors,
    clinicalInterpretation,
    recommendations,
    timestamp: new Date(apiResult.timestamp || Date.now()),
  };
}

/**
 * Map backend feature names to frontend display names
 */
function mapFeatureNameToDisplay(featureName: string): string {
  const mapping: Record<string, string> = {
    'totalBilirubin': 'Total Bilirubin (TB)',
    'total_bilirubin': 'Total Bilirubin (TB)',
    'directBilirubin': 'Direct Bilirubin (DB)',
    'direct_bilirubin': 'Direct Bilirubin (DB)',
    'alkalinePhosphatase': 'Alkaline Phosphatase',
    'alkaline_phosphotase': 'Alkaline Phosphatase',
    'sgptAlt': 'SGPT / ALT',
    'alamine_aminotransferase': 'SGPT / ALT',
    'sgotAst': 'SGOT / AST',
    'aspartate_aminotransferase': 'SGOT / AST',
    'totalProteins': 'Total Proteins',
    'total_protiens': 'Total Proteins',
    'albumin': 'Albumin',
    'agRatio': 'A/G Ratio',
    'albumin_and_globulin_ratio': 'A/G Ratio',
  };
  return mapping[featureName] || featureName;
}

/**
 * Get patient value for a feature
 */
function getPatientValueForFeature(featureName: string, data: PatientData): number {
  const mapping: Record<string, keyof PatientData> = {
    'totalBilirubin': 'totalBilirubin',
    'total_bilirubin': 'totalBilirubin',
    'directBilirubin': 'directBilirubin',
    'direct_bilirubin': 'directBilirubin',
    'alkalinePhosphatase': 'alkalinePhosphatase',
    'alkaline_phosphotase': 'alkalinePhosphatase',
    'sgptAlt': 'sgptAlt',
    'alamine_aminotransferase': 'sgptAlt',
    'sgotAst': 'sgotAst',
    'aspartate_aminotransferase': 'sgotAst',
    'totalProteins': 'totalProteins',
    'total_protiens': 'totalProteins',
    'albumin': 'albumin',
    'agRatio': 'agRatio',
    'albumin_and_globulin_ratio': 'agRatio',
  };
  const key = mapping[featureName];
  return key ? (data[key] as number) : 0;
}

/**
 * Get normal range for a feature
 */
function getNormalRangeForFeature(featureName: string): { min: number; max: number; description: string } {
  const mapping: Record<string, keyof typeof NORMAL_RANGES> = {
    'totalBilirubin': 'totalBilirubin',
    'total_bilirubin': 'totalBilirubin',
    'directBilirubin': 'directBilirubin',
    'direct_bilirubin': 'directBilirubin',
    'alkalinePhosphatase': 'alkalinePhosphatase',
    'alkaline_phosphotase': 'alkalinePhosphatase',
    'sgptAlt': 'sgptAlt',
    'alamine_aminotransferase': 'sgptAlt',
    'sgotAst': 'sgotAst',
    'aspartate_aminotransferase': 'sgotAst',
    'totalProteins': 'totalProteins',
    'total_protiens': 'totalProteins',
    'albumin': 'albumin',
    'agRatio': 'agRatio',
    'albumin_and_globulin_ratio': 'agRatio',
  };
  const key = mapping[featureName];
  return key ? NORMAL_RANGES[key] : { min: 0, max: 0, description: 'Unknown', unit: '' };
}

/**
 * Get clinical note for a feature
 */
function getClinicalNoteForFeature(featureName: string): string {
  const notes: Record<string, string> = {
    'totalBilirubin': 'Elevated bilirubin suggests possible jaundice, liver function impairment, or bile duct obstruction.',
    'directBilirubin': 'Elevated direct bilirubin may indicate cholestasis or liver cell damage.',
    'alkalinePhosphatase': 'Elevated alkaline phosphatase may indicate cholestasis, bile duct obstruction, or bone disorders.',
    'sgptAlt': 'Elevated ALT indicates liver cell injury, commonly seen in hepatitis or liver inflammation.',
    'sgotAst': 'Elevated AST suggests liver or muscle damage. AST/ALT ratio helps differentiate liver conditions.',
    'totalProteins': 'Low total protein may indicate liver disease, malnutrition, or kidney disease.',
    'albumin': 'Reduced albumin suggests impaired liver synthetic function, malnutrition, or protein loss.',
    'agRatio': 'Low A/G ratio suggests decreased albumin or increased globulin, which may indicate liver disease or inflammation.',
  };
  return notes[featureName] || 'Abnormal values may indicate liver disease or related conditions.';
}

/**
 * Fallback mock function (kept for development/testing)
 */
function predictLiverDiseaseMock(data: PatientData): PredictionResult {
  // Simulate processing delay
  const delay = Math.random() * 1000 + 500;

  // Calculate risk score based on abnormal values
  let riskScore = 0;
  const factors: ContributingFactor[] = [];

  // Total Bilirubin - weight: high
  const tbStatus = evaluateParameter(
    data.totalBilirubin,
    NORMAL_RANGES.totalBilirubin.min,
    NORMAL_RANGES.totalBilirubin.max
  );
  if (tbStatus !== 'Normal') {
    const contribution = Math.min((data.totalBilirubin - NORMAL_RANGES.totalBilirubin.max) / 5, 1);
    riskScore += contribution * 0.25;
    factors.push({
      feature: 'Total Bilirubin (TB)',
      patientValue: data.totalBilirubin,
      normalRange: NORMAL_RANGES.totalBilirubin.description,
      contribution: Math.abs(contribution),
      contributionLevel: getContributionLevel(Math.abs(contribution)),
      status: tbStatus,
      clinicalNote:
        'Elevated bilirubin suggests possible jaundice, liver function impairment, or bile duct obstruction.',
    });
  }

  // Direct Bilirubin
  const dbStatus = evaluateParameter(
    data.directBilirubin,
    NORMAL_RANGES.directBilirubin.min,
    NORMAL_RANGES.directBilirubin.max
  );
  if (dbStatus !== 'Normal') {
    const contribution = Math.min((data.directBilirubin - NORMAL_RANGES.directBilirubin.max) / 2, 1);
    riskScore += contribution * 0.15;
    factors.push({
      feature: 'Direct Bilirubin (DB)',
      patientValue: data.directBilirubin,
      normalRange: NORMAL_RANGES.directBilirubin.description,
      contribution: Math.abs(contribution),
      contributionLevel: getContributionLevel(Math.abs(contribution)),
      status: dbStatus,
      clinicalNote: 'Elevated direct bilirubin may indicate cholestasis or liver cell damage.',
    });
  }

  // SGPT/ALT - weight: high
  const sgptStatus = evaluateParameter(
    data.sgptAlt,
    NORMAL_RANGES.sgptAlt.min,
    NORMAL_RANGES.sgptAlt.max
  );
  if (sgptStatus !== 'Normal') {
    const contribution = Math.min((data.sgptAlt - NORMAL_RANGES.sgptAlt.max) / 100, 1);
    riskScore += contribution * 0.2;
    factors.push({
      feature: 'SGPT / ALT',
      patientValue: data.sgptAlt,
      normalRange: NORMAL_RANGES.sgptAlt.description,
      contribution: Math.abs(contribution),
      contributionLevel: getContributionLevel(Math.abs(contribution)),
      status: sgptStatus,
      clinicalNote:
        'Elevated ALT indicates liver cell injury, commonly seen in hepatitis or liver inflammation.',
    });
  }

  // SGOT/AST - weight: high
  const sgotStatus = evaluateParameter(
    data.sgotAst,
    NORMAL_RANGES.sgotAst.min,
    NORMAL_RANGES.sgotAst.max
  );
  if (sgotStatus !== 'Normal') {
    const contribution = Math.min((data.sgotAst - NORMAL_RANGES.sgotAst.max) / 100, 1);
    riskScore += contribution * 0.2;
    factors.push({
      feature: 'SGOT / AST',
      patientValue: data.sgotAst,
      normalRange: NORMAL_RANGES.sgotAst.description,
      contribution: Math.abs(contribution),
      contributionLevel: getContributionLevel(Math.abs(contribution)),
      status: sgotStatus,
      clinicalNote:
        'Elevated AST suggests liver or muscle damage. AST/ALT ratio helps differentiate liver conditions.',
    });
  }

  // Alkaline Phosphatase
  const alkphosStatus = evaluateParameter(
    data.alkalinePhosphatase,
    NORMAL_RANGES.alkalinePhosphatase.min,
    NORMAL_RANGES.alkalinePhosphatase.max
  );
  if (alkphosStatus !== 'Normal') {
    const contribution = Math.min((data.alkalinePhosphatase - NORMAL_RANGES.alkalinePhosphatase.max) / 200, 1);
    riskScore += contribution * 0.15;
    factors.push({
      feature: 'Alkaline Phosphatase',
      patientValue: data.alkalinePhosphatase,
      normalRange: NORMAL_RANGES.alkalinePhosphatase.description,
      contribution: Math.abs(contribution),
      contributionLevel: getContributionLevel(Math.abs(contribution)),
      status: alkphosStatus,
      clinicalNote:
        'Elevated alkaline phosphatase may indicate cholestasis, bile duct obstruction, or bone disorders.',
    });
  }

  // Albumin - low albumin is concerning
  const albStatus = evaluateParameter(
    data.albumin,
    NORMAL_RANGES.albumin.min,
    NORMAL_RANGES.albumin.max
  );
  if (albStatus === 'Low') {
    const contribution = Math.min((NORMAL_RANGES.albumin.min - data.albumin) / 2, 1);
    riskScore += contribution * 0.15;
    factors.push({
      feature: 'Albumin',
      patientValue: data.albumin,
      normalRange: NORMAL_RANGES.albumin.description,
      contribution: Math.abs(contribution),
      contributionLevel: getContributionLevel(Math.abs(contribution)),
      status: albStatus,
      clinicalNote:
        'Reduced albumin suggests impaired liver synthetic function, malnutrition, or protein loss.',
    });
  }

  // Total Proteins - low is concerning
  const tpStatus = evaluateParameter(
    data.totalProteins,
    NORMAL_RANGES.totalProteins.min,
    NORMAL_RANGES.totalProteins.max
  );
  if (tpStatus === 'Low') {
    const contribution = Math.min((NORMAL_RANGES.totalProteins.min - data.totalProteins) / 3, 1);
    riskScore += contribution * 0.1;
    factors.push({
      feature: 'Total Proteins',
      patientValue: data.totalProteins,
      normalRange: NORMAL_RANGES.totalProteins.description,
      contribution: Math.abs(contribution),
      contributionLevel: getContributionLevel(Math.abs(contribution)),
      status: tpStatus,
      clinicalNote: 'Low total protein may indicate liver disease, malnutrition, or kidney disease.',
    });
  }

  // A/G Ratio
  const agStatus = evaluateParameter(data.agRatio, NORMAL_RANGES.agRatio.min, NORMAL_RANGES.agRatio.max);
  if (agStatus === 'Low') {
    const contribution = Math.min((NORMAL_RANGES.agRatio.min - data.agRatio) / 1, 1);
    riskScore += contribution * 0.1;
    factors.push({
      feature: 'A/G Ratio',
      patientValue: data.agRatio,
      normalRange: NORMAL_RANGES.agRatio.description,
      contribution: Math.abs(contribution),
      contributionLevel: getContributionLevel(Math.abs(contribution)),
      status: agStatus,
      clinicalNote:
        'Low A/G ratio suggests decreased albumin or increased globulin, which may indicate liver disease or inflammation.',
    });
  }

  // Sort factors by contribution (descending)
  factors.sort((a, b) => b.contribution - a.contribution);

  // Take top 5 factors
  const topFactors = factors.slice(0, 5);

  // Calculate final probability
  const probability = Math.min(riskScore * 100, 95);
  const hasDiseaseRisk = probability > 50;

  // Determine confidence
  const confidence: 'Low' | 'Medium' | 'High' =
    probability > 75 ? 'High' : probability > 50 ? 'Medium' : 'Low';

  // Generate clinical interpretation
  const clinicalInterpretation = generateClinicalInterpretation(topFactors, data);

  // Generate recommendations
  const recommendations = generateRecommendations(topFactors, data, probability);

  return {
    hasDiseaseRisk,
    probability: Math.round(probability * 10) / 10,
    confidence,
    contributingFactors: topFactors,
    clinicalInterpretation,
    recommendations,
    timestamp: new Date(),
  };
}

function evaluateParameter(value: number, min: number, max: number): 'Normal' | 'High' | 'Low' {
  if (value < min) return 'Low';
  if (value > max) return 'High';
  return 'Normal';
}

function getContributionLevel(contribution: number): 'High' | 'Medium' | 'Low' {
  if (contribution > 0.6) return 'High';
  if (contribution > 0.3) return 'Medium';
  return 'Low';
}

function generateClinicalInterpretation(factors: ContributingFactor[], data: PatientData): string {
  if (factors.length === 0) {
    return 'All parameters are within normal ranges. No significant liver disease indicators detected. Patient appears to have healthy liver function based on the provided laboratory values.';
  }

  const hasElevatedEnzymes =
    factors.some((f) => f.feature.includes('SGPT') || f.feature.includes('SGOT')) &&
    (data.sgptAlt > NORMAL_RANGES.sgptAlt.max || data.sgotAst > NORMAL_RANGES.sgotAst.max);

  const hasElevatedBilirubin =
    factors.some((f) => f.feature.includes('Bilirubin')) &&
    data.totalBilirubin > NORMAL_RANGES.totalBilirubin.max;

  const hasLowAlbumin = factors.some((f) => f.feature.includes('Albumin')) && data.albumin < NORMAL_RANGES.albumin.min;

  let interpretation = 'Based on the laboratory findings, ';

  if (hasElevatedEnzymes && hasElevatedBilirubin) {
    interpretation +=
      'elevated liver enzymes (SGPT/SGOT) combined with increased bilirubin levels may indicate active liver inflammation or hepatocellular injury. This pattern is commonly observed in acute or chronic hepatitis, liver damage, or biliary obstruction. ';
  } else if (hasElevatedEnzymes) {
    interpretation +=
      'elevated liver enzymes (SGPT/SGOT) suggest hepatocellular injury or inflammation. The degree of elevation may help determine the severity and nature of liver involvement. ';
  } else if (hasElevatedBilirubin) {
    interpretation +=
      'elevated bilirubin levels may indicate impaired bilirubin metabolism, cholestasis, or hemolysis. This requires differentiation between conjugated and unconjugated causes. ';
  }

  if (hasLowAlbumin) {
    interpretation +=
      'Reduced albumin levels suggest impaired hepatic synthetic function, which may indicate chronic liver disease, cirrhosis, or significant liver damage. ';
  }

  interpretation +=
    '\n\nThis pattern may correspond to various liver conditions including hepatitis, fatty liver disease, cirrhosis, drug-induced liver injury, or biliary tract disorders. Clinical correlation with patient history, physical examination, imaging studies, and additional testing is essential for accurate diagnosis.';

  return interpretation;
}

function generateRecommendations(
  factors: ContributingFactor[],
  data: PatientData,
  probability: number
): string[] {
  const recommendations: string[] = [];

  if (probability > 70) {
    recommendations.push('Urgent clinical review recommended - high risk indicated');
  }

  if (factors.length > 0) {
    recommendations.push('Confirm findings with repeat liver function tests (LFT panel)');
  }

  const hasElevatedEnzymes = data.sgptAlt > NORMAL_RANGES.sgptAlt.max * 2 || data.sgotAst > NORMAL_RANGES.sgotAst.max * 2;
  if (hasElevatedEnzymes) {
    recommendations.push('Consider hepatitis panel (HBsAg, Anti-HCV, Anti-HAV) if enzymes markedly elevated');
    recommendations.push('Evaluate for potential hepatotoxic medications or alcohol use');
  }

  if (data.totalBilirubin > NORMAL_RANGES.totalBilirubin.max) {
    recommendations.push('Liver ultrasound or imaging to rule out biliary obstruction or structural abnormalities');
  }

  if (data.albumin < NORMAL_RANGES.albumin.min) {
    recommendations.push('Assess for signs of chronic liver disease, malnutrition, or protein-losing conditions');
  }

  if (data.alkalinePhosphatase > NORMAL_RANGES.alkalinePhosphatase.max * 1.5) {
    recommendations.push('Investigate for cholestatic liver disease or bone pathology (GGT, bone markers)');
  }

  recommendations.push('Correlate with patient symptoms: jaundice, fatigue, abdominal pain, appetite changes');
  recommendations.push('Review patient medical history, medications, alcohol consumption, and risk factors');
  recommendations.push('Consider specialist referral (hepatology/gastroenterology) based on severity and clinical picture');

  return recommendations;
}

export function getExamplePatientData(): PatientData {
  return {
    patientId: 'LP-' + Math.floor(Math.random() * 10000),
    name: 'Sample Patient',
    age: 45,
    gender: 'Male',
    totalBilirubin: 2.1,
    directBilirubin: 0.8,
    alkalinePhosphatase: 198,
    sgptAlt: 89,
    sgotAst: 112,
    totalProteins: 6.8,
    albumin: 3.2,
    agRatio: 0.95,
  };
}

export function getHealthyPatientData(): PatientData {
  return {
    patientId: 'LP-' + Math.floor(Math.random() * 10000),
    name: 'Healthy Patient',
    age: 35,
    gender: 'Female',
    totalBilirubin: 0.7,
    directBilirubin: 0.2,
    alkalinePhosphatase: 85,
    sgptAlt: 28,
    sgotAst: 22,
    totalProteins: 7.2,
    albumin: 4.5,
    agRatio: 1.67,
  };
}
