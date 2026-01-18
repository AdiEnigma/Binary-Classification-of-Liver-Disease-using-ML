import { TrendingUp, ArrowUp, ArrowDown, Minus, Brain } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { Progress } from '@/app/components/ui/progress';
import { ContributingFactor } from '@/app/types/patient';

interface ContributingFactorsProps {
  factors: ContributingFactor[];
  showFactors: boolean;
}

export function ContributingFactors({ factors, showFactors }: ContributingFactorsProps) {
  if (!showFactors || factors.length === 0) {
    return null;
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'High':
        return <ArrowUp className="w-4 h-4 text-red-600" />;
      case 'Low':
        return <ArrowDown className="w-4 h-4 text-blue-600" />;
      default:
        return <Minus className="w-4 h-4 text-green-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'High':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'Low':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      default:
        return 'text-green-600 bg-green-50 border-green-200';
    }
  };

  const getContributionColor = (level: string) => {
    switch (level) {
      case 'High':
        return 'bg-red-500';
      case 'Medium':
        return 'bg-amber-500';
      default:
        return 'bg-yellow-500';
    }
  };

  return (
    <Card className="border-2 border-amber-200">
      <CardHeader className="bg-gradient-to-r from-amber-50 to-orange-50 border-b">
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-amber-600" />
          Key Contributing Factors
        </CardTitle>
        <CardDescription>
          Model explanation: Features that most influenced the prediction (SHAP-based analysis)
        </CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-4">
          {factors.map((factor, index) => (
            <div
              key={index}
              className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
            >
              <div className="space-y-3">
                {/* Factor Header */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold text-gray-900">{factor.feature}</h4>
                      <Badge variant="outline" className={getStatusColor(factor.status)}>
                        {getStatusIcon(factor.status)}
                        <span className="ml-1">{factor.status}</span>
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600">{factor.normalRange}</p>
                  </div>

                  <div className="text-right">
                    <p className="text-2xl font-bold text-gray-900">
                      {factor.patientValue.toFixed(2)}
                    </p>
                    <Badge
                      variant="secondary"
                      className={`mt-1 ${
                        factor.contributionLevel === 'High'
                          ? 'bg-red-100 text-red-700'
                          : factor.contributionLevel === 'Medium'
                          ? 'bg-amber-100 text-amber-700'
                          : 'bg-yellow-100 text-yellow-700'
                      }`}
                    >
                      {factor.contributionLevel} Impact
                    </Badge>
                  </div>
                </div>

                {/* Contribution Bar */}
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-gray-600">
                    <span>Contribution to Prediction</span>
                    <span className="font-semibold">{(factor.contribution * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${getContributionColor(factor.contributionLevel)} transition-all`}
                      style={{ width: `${factor.contribution * 100}%` }}
                    />
                  </div>
                </div>

                {/* Clinical Note */}
                <div className="bg-blue-50 border-l-4 border-blue-400 p-3 rounded">
                  <p className="text-sm text-blue-900">
                    <span className="font-semibold">Clinical Note: </span>
                    {factor.clinicalNote}
                  </p>
                </div>
              </div>
            </div>
          ))}

          {/* Explanation Footer */}
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mt-6">
            <div className="flex items-start gap-3">
              <TrendingUp className="w-5 h-5 text-gray-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm font-semibold text-gray-900 mb-1">
                  Understanding Model Explainability
                </p>
                <p className="text-xs text-gray-600">
                  The factors above are ranked by their contribution to the model's prediction. Features with
                  higher contribution have a stronger influence on the final risk assessment. This analysis
                  uses techniques similar to SHAP (SHapley Additive exPlanations) to provide interpretable
                  insights into the model's decision-making process.
                </p>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
