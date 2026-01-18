import { CheckSquare, FileDown, Copy, Save, Stethoscope } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/app/components/ui/card';
import { Button } from '@/app/components/ui/button';
import { Badge } from '@/app/components/ui/badge';
import { toast } from 'sonner';
import { PredictionResult, PatientData } from '@/app/types/patient';

interface RecommendationsProps {
  recommendations: string[];
  result: PredictionResult;
  patientData: PatientData;
}

export function Recommendations({ recommendations, result, patientData }: RecommendationsProps) {
  const handleCopySummary = () => {
    const summary = generateSummaryText();
    navigator.clipboard.writeText(summary);
    toast.success('Summary copied to clipboard!');
  };

  const handleExportPDF = () => {
    // In a real application, this would generate a PDF
    toast.info('PDF export functionality would be implemented here');
  };

  const handleSaveToRecord = () => {
    // In a real application, this would save to a database
    toast.info('Save to patient record functionality would be implemented here');
  };

  const generateSummaryText = () => {
    let text = '=== LIVER DISEASE RISK PREDICTION REPORT ===\n\n';
    text += `Patient ID: ${patientData.patientId || 'N/A'}\n`;
    text += `Name: ${patientData.name || 'N/A'}\n`;
    text += `Age: ${patientData.age} | Gender: ${patientData.gender}\n`;
    text += `Date: ${result.timestamp.toLocaleString()}\n\n`;

    text += '--- PREDICTION SUMMARY ---\n';
    text += `Risk Status: ${result.hasDiseaseRisk ? 'DISEASE RISK DETECTED' : 'NO DISEASE DETECTED'}\n`;
    text += `Risk Probability: ${result.probability}%\n`;
    text += `Model Confidence: ${result.confidence}\n\n`;

    if (result.contributingFactors.length > 0) {
      text += '--- KEY CONTRIBUTING FACTORS ---\n';
      result.contributingFactors.forEach((factor, i) => {
        text += `${i + 1}. ${factor.feature}: ${factor.patientValue} (${factor.status})\n`;
        text += `   ${factor.normalRange}\n`;
        text += `   ${factor.clinicalNote}\n\n`;
      });
    }

    text += '--- CLINICAL INTERPRETATION ---\n';
    text += `${result.clinicalInterpretation}\n\n`;

    text += '--- RECOMMENDATIONS ---\n';
    recommendations.forEach((rec, i) => {
      text += `${i + 1}. ${rec}\n`;
    });

    text += '\n--- DISCLAIMER ---\n';
    text += 'This is an ML-based risk prediction tool. Final diagnosis must be made by a qualified physician.\n';

    return text;
  };

  return (
    <Card className="border-2 border-green-200">
      <CardHeader className="bg-gradient-to-r from-green-50 to-teal-50 border-b">
        <CardTitle className="flex items-center gap-2">
          <Stethoscope className="w-5 h-5 text-green-600" />
          Clinical Recommendations
        </CardTitle>
        <CardDescription>Recommended follow-up actions for physician review</CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Recommendations Checklist */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 mb-3">
              <CheckSquare className="w-5 h-5 text-green-600" />
              <h4 className="font-semibold text-gray-900">Recommended Follow-ups</h4>
            </div>

            <div className="space-y-2">
              {recommendations.map((recommendation, index) => (
                <div
                  key={index}
                  className="flex items-start gap-3 p-3 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex-shrink-0 mt-0.5">
                    <div className="w-5 h-5 border-2 border-green-600 rounded flex items-center justify-center">
                      <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-700 flex-1">{recommendation}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Urgency Badge */}
          {result.probability > 70 && (
            <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-r">
              <div className="flex items-center gap-2">
                <Badge variant="destructive" className="bg-red-600">
                  HIGH PRIORITY
                </Badge>
                <p className="text-sm text-red-900 font-semibold">
                  Urgent clinical review recommended based on high risk probability
                </p>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="pt-4 border-t space-y-3">
            <h4 className="text-sm font-semibold text-gray-700 mb-2">Report Actions</h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopySummary}
                className="w-full"
              >
                <Copy className="w-4 h-4 mr-2" />
                Copy Summary
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={handleExportPDF}
                className="w-full"
              >
                <FileDown className="w-4 h-4 mr-2" />
                Export PDF
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={handleSaveToRecord}
                className="w-full"
              >
                <Save className="w-4 h-4 mr-2" />
                Save to Record
              </Button>
            </div>
          </div>

          {/* Final Note */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-xs text-blue-900">
              <span className="font-semibold">Note for Healthcare Providers:</span> These recommendations
              are generated based on the laboratory findings and prediction results. Please integrate them
              with comprehensive patient assessment, including clinical history, physical examination, imaging
              studies, and your professional judgment. Consider patient-specific factors such as comorbidities,
              medications, and risk factors when determining the appropriate diagnostic and therapeutic approach.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
