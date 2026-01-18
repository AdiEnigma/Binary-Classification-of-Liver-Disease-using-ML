import { AlertCircle, CheckCircle2, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { Progress } from '@/app/components/ui/progress';
import { PredictionResult } from '@/app/types/patient';

interface PredictionSummaryProps {
  result: PredictionResult;
}

export function PredictionSummary({ result }: PredictionSummaryProps) {
  const { hasDiseaseRisk, probability, confidence } = result;

  return (
    <Card className="border-2">
      <CardHeader className={hasDiseaseRisk ? 'bg-red-50 border-b' : 'bg-green-50 border-b'}>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            {hasDiseaseRisk ? (
              <AlertCircle className="w-6 h-6 text-red-600" />
            ) : (
              <CheckCircle2 className="w-6 h-6 text-green-600" />
            )}
            Prediction Summary
          </span>

          <Badge
            variant={hasDiseaseRisk ? 'destructive' : 'default'}
            className={
              hasDiseaseRisk
                ? 'bg-red-600 text-white'
                : 'bg-green-600 text-white hover:bg-green-700'
            }
          >
            {confidence} Confidence
          </Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Status Badge */}
          <div className="flex justify-center">
            <div
              className={`px-8 py-4 rounded-xl text-center border shadow-sm ${
                hasDiseaseRisk
                  ? 'bg-red-600 border-red-200'
                  : 'bg-green-600 border-green-200'
              }`}
            >
              <p className="text-sm text-white/90 mb-1">Model Prediction</p>
              <h3 className="text-2xl font-bold text-white">
                {hasDiseaseRisk ? 'Liver Disease Risk Detected' : 'No Liver Disease Detected'}
              </h3>
            </div>
          </div>

          {/* Risk Probability */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingUp className={`w-5 h-5 ${hasDiseaseRisk ? 'text-red-600' : 'text-green-600'}`} />
                <span className="font-semibold text-gray-700">Risk Probability</span>
              </div>

              <span
                className={`text-3xl font-bold ${
                  hasDiseaseRisk ? 'text-red-600' : 'text-green-600'
                }`}
              >
                {probability.toFixed(1)}%
              </span>
            </div>

            <Progress
              value={probability}
              className={`h-3 ${
                hasDiseaseRisk ? '[&>div]:bg-red-600' : '[&>div]:bg-green-600'
              }`}
            />

            <p className="text-sm text-gray-600 text-center">
              {probability < 30 && 'Low risk of liver disease based on current parameters'}
              {probability >= 30 && probability < 60 && 'Moderate risk - some parameters are abnormal'}
              {probability >= 60 && probability < 80 && 'High risk - multiple abnormal parameters detected'}
              {probability >= 80 && 'Very high risk - significant abnormalities in liver function tests'}
            </p>
          </div>

          {/* Confidence Indicator */}
          <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
            <div className="flex items-start gap-3">
              <div className="flex-1">
                <p className="text-sm font-semibold text-gray-700 mb-1">
                  Model Confidence: {confidence}
                </p>
                <p className="text-xs text-gray-600">
                  {confidence === 'High' &&
                    'The model has high confidence in this prediction based on clear patterns in the data.'}
                  {confidence === 'Medium' &&
                    'The model has moderate confidence. Additional clinical evaluation is recommended.'}
                  {confidence === 'Low' &&
                    'The model has lower confidence. Results should be interpreted cautiously with clinical context.'}
                </p>
              </div>
            </div>
          </div>

          {/* Timestamp */}
          <div className="text-center pt-2 border-t">
            <p className="text-xs text-gray-500">
              Prediction generated: {result.timestamp.toLocaleString()}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
