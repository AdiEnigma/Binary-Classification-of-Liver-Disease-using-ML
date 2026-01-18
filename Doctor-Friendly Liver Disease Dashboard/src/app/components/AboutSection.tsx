import { Database, Brain, BarChart3, Shield, FileText, Activity, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';

export function AboutSection() {
  return (
    <div className="max-w-5xl mx-auto py-8 px-4 space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">About the Liver Disease Prediction System</h2>
        <p className="text-gray-600">
          AI-powered clinical decision support tool for liver disease risk assessment
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Dataset Information */}
        <Card>
          <CardHeader className="bg-gradient-to-r from-blue-50 to-cyan-50 border-b">
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-600" />
              Dataset: ILPD
            </CardTitle>
            <CardDescription>Indian Liver Patient Dataset</CardDescription>
          </CardHeader>
          <CardContent className="p-6 space-y-3">
            <p className="text-sm text-gray-700">
              The system is trained on the <strong>Indian Liver Patient Dataset (ILPD)</strong>, which contains
              laboratory test results from liver patients and healthy individuals.
            </p>
            <div className="bg-blue-50 p-3 rounded-lg">
              <p className="text-xs text-blue-900 font-semibold mb-1">Dataset Features:</p>
              <ul className="text-xs text-blue-800 space-y-1">
                <li>• Total Bilirubin (TB)</li>
                <li>• Direct Bilirubin (DB)</li>
                <li>• Alkaline Phosphatase</li>
                <li>• SGPT / ALT (Alamine Aminotransferase)</li>
                <li>• SGOT / AST (Aspartate Aminotransferase)</li>
                <li>• Total Proteins</li>
                <li>• Albumin</li>
                <li>• Albumin/Globulin Ratio</li>
              </ul>
            </div>
            <p className="text-xs text-gray-600">
              These features represent standard liver function tests commonly used in clinical practice.
            </p>
          </CardContent>
        </Card>

        {/* Model Information */}
        <Card>
          <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b">
            <CardTitle className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-600" />
              Machine Learning Model
            </CardTitle>
            <CardDescription>AI Algorithm Details</CardDescription>
          </CardHeader>
          <CardContent className="p-6 space-y-3">
            <p className="text-sm text-gray-700">
              The prediction system uses advanced <strong>supervised machine learning algorithms</strong>{' '}
              trained on historical patient data to identify patterns associated with liver disease.
            </p>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-300">
                  Classification
                </Badge>
                <span className="text-xs text-gray-600">Binary prediction (Disease/No Disease)</span>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-300">
                  Algorithms
                </Badge>
                <span className="text-xs text-gray-600">Random Forest / Logistic Regression / XGBoost</span>
              </div>
            </div>
            <div className="bg-purple-50 p-3 rounded-lg">
              <p className="text-xs text-purple-900">
                <strong>Model Training:</strong> The model is trained using cross-validation techniques to
                ensure robust performance across diverse patient populations.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Explainability */}
        <Card>
          <CardHeader className="bg-gradient-to-r from-amber-50 to-orange-50 border-b">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-amber-600" />
              Model Explainability
            </CardTitle>
            <CardDescription>Understanding AI Decisions</CardDescription>
          </CardHeader>
          <CardContent className="p-6 space-y-3">
            <p className="text-sm text-gray-700">
              The system uses <strong>SHAP (SHapley Additive exPlanations)</strong> and{' '}
              <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong> techniques to provide
              transparent insights into predictions.
            </p>
            <div className="bg-amber-50 p-3 rounded-lg space-y-2">
              <p className="text-xs text-amber-900 font-semibold">Key Benefits:</p>
              <ul className="text-xs text-amber-800 space-y-1">
                <li>✓ Identifies which features most influenced the prediction</li>
                <li>✓ Quantifies the contribution of each laboratory parameter</li>
                <li>✓ Helps clinicians understand the "why" behind predictions</li>
                <li>✓ Builds trust in AI-assisted decision making</li>
              </ul>
            </div>
            <p className="text-xs text-gray-600">
              This transparency is crucial for clinical adoption and helps physicians validate AI recommendations
              against their domain expertise.
            </p>
          </CardContent>
        </Card>

        {/* Clinical Usage */}
        <Card>
          <CardHeader className="bg-gradient-to-r from-green-50 to-teal-50 border-b">
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-600" />
              Clinical Usage Guidelines
            </CardTitle>
            <CardDescription>Best Practices for Healthcare Providers</CardDescription>
          </CardHeader>
          <CardContent className="p-6 space-y-3">
            <p className="text-sm text-gray-700">
              This tool is designed as a <strong>clinical decision support system</strong>, not a replacement
              for medical judgment.
            </p>
            <div className="space-y-2">
              <div className="bg-green-50 p-3 rounded-lg">
                <p className="text-xs text-green-900 font-semibold mb-1">✓ Appropriate Uses:</p>
                <ul className="text-xs text-green-800 space-y-1">
                  <li>• Screening and risk stratification</li>
                  <li>• Supplement to clinical evaluation</li>
                  <li>• Identifying patients needing further workup</li>
                  <li>• Educational tool for medical trainees</li>
                </ul>
              </div>
              <div className="bg-red-50 p-3 rounded-lg">
                <p className="text-xs text-red-900 font-semibold mb-1">✗ Not Suitable For:</p>
                <ul className="text-xs text-red-800 space-y-1">
                  <li>• Standalone diagnosis without clinical correlation</li>
                  <li>• Emergency or critical care decisions</li>
                  <li>• Replacing comprehensive patient evaluation</li>
                  <li>• Legal or regulatory documentation</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Important Information Banner */}
      <Card className="border-2 border-amber-300 bg-amber-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-amber-900">
            <AlertCircle className="w-6 h-6 text-amber-600" />
            Important Information & Limitations
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-2 text-sm text-amber-900">
            <p>
              <strong>Model Performance:</strong> While the model demonstrates good performance on validation
              data, real-world clinical scenarios may vary. The model's accuracy depends on:
            </p>
            <ul className="list-disc list-inside ml-4 space-y-1 text-xs">
              <li>Quality and accuracy of input laboratory values</li>
              <li>Patient population characteristics matching the training dataset</li>
              <li>Proper interpretation within appropriate clinical context</li>
            </ul>

            <p className="pt-2">
              <strong>Known Limitations:</strong>
            </p>
            <ul className="list-disc list-inside ml-4 space-y-1 text-xs">
              <li>Does not account for patient symptoms, history, or physical examination findings</li>
              <li>Cannot detect specific liver disease subtypes (viral, alcoholic, autoimmune, etc.)</li>
              <li>May not generalize well to populations significantly different from ILPD dataset</li>
              <li>Cannot replace imaging studies or liver biopsy when indicated</li>
            </ul>

            <p className="pt-2">
              <strong>Data Privacy:</strong> This demonstration system does not store or transmit patient data.
              In production environments, appropriate HIPAA-compliant data handling procedures must be
              implemented.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Technical Specifications */}
      <Card>
        <CardHeader className="bg-gradient-to-r from-gray-50 to-slate-50 border-b">
          <CardTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-gray-600" />
            Technical Specifications
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">System Architecture</h4>
              <ul className="space-y-1 text-xs text-gray-700">
                <li>• Frontend: React + TypeScript</li>
                <li>• Styling: Tailwind CSS v4</li>
                <li>• Charts: Recharts library</li>
                <li>• Animations: Motion (Framer Motion)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">ML Integration</h4>
              <ul className="space-y-1 text-xs text-gray-700">
                <li>• Model: Classification (Binary)</li>
                <li>• Explainability: SHAP/LIME-based</li>
                <li>• Output: Risk probability + factors</li>
                <li>• Processing: Real-time inference</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Contact/Support */}
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-6 text-center">
        <Shield className="w-8 h-8 text-teal-600 mx-auto mb-2" />
        <h4 className="font-semibold text-gray-900 mb-1">For Healthcare Professionals</h4>
        <p className="text-sm text-gray-700">
          This system is intended for use by licensed medical professionals, laboratory staff, and clinical
          researchers. For technical support or to report issues, please contact your system administrator.
        </p>
      </div>
    </div>
  );
}
