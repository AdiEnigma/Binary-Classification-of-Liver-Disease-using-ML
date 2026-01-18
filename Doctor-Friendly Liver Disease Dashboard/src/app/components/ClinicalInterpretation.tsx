import { FileText, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/app/components/ui/card';

interface ClinicalInterpretationProps {
  interpretation: string;
}

export function ClinicalInterpretation({ interpretation }: ClinicalInterpretationProps) {
  return (
    <Card className="border-2">
      <CardHeader className="bg-gradient-to-r from-blue-50 to-cyan-50 border-b">
        <CardTitle className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-600" />
          Clinical Interpretation
        </CardTitle>
        <CardDescription>Integrated analysis of laboratory findings</CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-4">
          {/* Warning Banner */}
          <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded-r">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm font-semibold text-amber-900 mb-1">
                  Supportive Information Only
                </p>
                <p className="text-xs text-amber-800">
                  This interpretation provides possible clinical correlations based on laboratory patterns.
                  It is not a definitive diagnosis and must be validated by a qualified physician through
                  comprehensive clinical assessment.
                </p>
              </div>
            </div>
          </div>

          {/* Interpretation Text */}
          <div className="prose prose-sm max-w-none">
            <div className="bg-white border border-gray-200 rounded-lg p-5">
              <p className="text-gray-800 leading-relaxed whitespace-pre-line">{interpretation}</p>
            </div>
          </div>

          {/* Additional Context */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-900">
              <span className="font-semibold">Important: </span>
              This interpretation uses terms such as "may indicate," "suggests," and "possible" to reflect
              that laboratory findings must be considered alongside patient history, symptoms, physical
              examination, imaging, and other diagnostic information. Liver disease encompasses a wide
              spectrum of conditions with varying etiologies, requiring expert clinical judgment for accurate
              diagnosis and management.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
