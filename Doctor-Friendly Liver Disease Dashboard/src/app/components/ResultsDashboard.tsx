import { motion } from 'motion/react';
import { Activity, AlertCircle } from 'lucide-react';
import { PredictionSummary } from './PredictionSummary';
import { ContributingFactors } from './ContributingFactors';
import { ClinicalInterpretation } from './ClinicalInterpretation';
import { VisualCharts } from './VisualCharts';
import { Recommendations } from './Recommendations';
import { PatientData, PredictionResult } from '@/app/types/patient';

interface ResultsDashboardProps {
  result: PredictionResult | null;
  patientData: PatientData | null;
}

export function ResultsDashboard({ result, patientData }: ResultsDashboardProps) {
  if (!result || !patientData) {
    return (
      <div className="h-full flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg border-2 border-dashed border-gray-300">
        <div className="text-center p-8 max-w-md">
          <div className="w-20 h-20 bg-gradient-to-br from-teal-100 to-cyan-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Activity className="w-10 h-10 text-teal-600" />
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">Ready to Analyze</h3>
          <p className="text-gray-600 mb-4">
            Enter patient laboratory values in the form and click "Predict Risk" to generate AI-powered
            liver disease risk assessment and clinical insights.
          </p>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
            <p className="text-xs text-amber-800 text-left">
              Remember: This tool provides decision support only. All predictions must be validated by
              qualified medical professionals.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const showFactors = result.hasDiseaseRisk || result.probability > 50;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6 overflow-y-auto max-h-[calc(100vh-140px)] pb-6"
    >
      {/* Prediction Summary */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
      >
        <PredictionSummary result={result} />
      </motion.div>

      {/* Contributing Factors */}
      {showFactors && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
        >
          <ContributingFactors factors={result.contributingFactors} showFactors={showFactors} />
        </motion.div>
      )}

      {/* Clinical Interpretation */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
      >
        <ClinicalInterpretation interpretation={result.clinicalInterpretation} />
      </motion.div>

      {/* Visual Charts */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.4 }}
      >
        <VisualCharts patientData={patientData} factors={result.contributingFactors} />
      </motion.div>

      {/* Recommendations */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.5 }}
      >
        <Recommendations
          recommendations={result.recommendations}
          result={result}
          patientData={patientData}
        />
      </motion.div>
    </motion.div>
  );
}
