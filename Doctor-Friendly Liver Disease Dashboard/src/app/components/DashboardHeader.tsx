import { Activity, Home, HelpCircle, Upload, ClipboardList, AlertCircle } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/app/components/ui/dialog';
import { Badge } from '@/app/components/ui/badge';

type ViewMode = 'home' | 'predict' | 'bulk' | 'help';

interface DashboardHeaderProps {
  currentView: ViewMode;
  onViewChange: (view: ViewMode) => void;
}

export function DashboardHeader({ currentView, onViewChange }: DashboardHeaderProps) {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Top Bar */}
        <div className="flex items-center justify-between py-4">
          {/* Logo and Title */}
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl">
              <Activity className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">
                Liver Disease Prediction System
              </h1>
              <p className="text-sm text-gray-600">
                ILPD ML-Based Decision Support Dashboard for Clinical Use
              </p>
            </div>
          </div>

          {/* Badge only (Date/Time removed) */}
          <div className="flex items-center gap-4">
            <Badge
              variant="outline"
              className="px-3 py-1 bg-teal-50 border-teal-300 text-teal-700"
            >
              <Activity className="w-3 h-3 mr-1" />
              Clinical System
            </Badge>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex items-center justify-between pb-3 border-t border-gray-100 pt-3">
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              variant={currentView === 'home' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => onViewChange('home')}
              className={currentView === 'home' ? 'bg-teal-600 hover:bg-teal-700' : ''}
            >
              <Home className="w-4 h-4 mr-2" />
              Home
            </Button>

            <Button
              variant={currentView === 'predict' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => onViewChange('predict')}
              className={currentView === 'predict' ? 'bg-teal-600 hover:bg-teal-700' : ''}
              title="Individual patient check"
            >
              <ClipboardList className="w-4 h-4 mr-2" />
              Individual Check
            </Button>

            <Button
              variant={currentView === 'bulk' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => onViewChange('bulk')}
              className={currentView === 'bulk' ? 'bg-teal-600 hover:bg-teal-700' : ''}
              title="Upload CSV for batch prediction"
            >
              <Upload className="w-4 h-4 mr-2" />
              Bulk CSV
            </Button>

            <Button
              variant={currentView === 'help' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => onViewChange('help')}
              className={currentView === 'help' ? 'bg-teal-600 hover:bg-teal-700' : ''}
            >
              <HelpCircle className="w-4 h-4 mr-2" />
              Help
            </Button>
          </div>

          <Dialog>
            <DialogTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="border-amber-300 bg-amber-50 text-amber-800 hover:bg-amber-100"
              >
                <AlertCircle className="w-4 h-4 mr-2" />
                Important Disclaimer
              </Button>
            </DialogTrigger>

            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2 text-xl">
                  <AlertCircle className="w-6 h-6 text-amber-600" />
                  Medical Disclaimer
                </DialogTitle>

                <DialogDescription className="text-base space-y-4 pt-4">
                  <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded">
                    <p className="font-semibold text-amber-900 mb-2">
                      This tool provides ML-based risk prediction support. It is NOT a medical diagnosis.
                    </p>
                  </div>

                  <div className="space-y-3 text-gray-700">
                    <p>
                      <strong>Important Information:</strong>
                    </p>

                    <ul className="list-disc list-inside space-y-2 ml-2">
                      <li>
                        This system uses machine learning algorithms trained on the Indian Liver Patient Dataset
                        (ILPD) to provide risk assessment support only.
                      </li>
                      <li>
                        The predictions and interpretations generated by this tool should be used as
                        supplementary information alongside clinical judgment.
                      </li>
                      <li>
                        <strong>
                          Final diagnosis and treatment decisions must be made by qualified healthcare
                          professionals
                        </strong>{' '}
                        after comprehensive clinical evaluation, patient history, physical examination, and
                        additional diagnostic testing.
                      </li>
                      <li>
                        This tool is not a substitute for professional medical advice, diagnosis, or treatment.
                      </li>
                      <li>
                        Always seek the advice of qualified physicians or other healthcare providers with any
                        questions regarding a medical condition.
                      </li>
                      <li>
                        The accuracy of predictions depends on the quality and completeness of input data.
                      </li>
                    </ul>

                    <div className="bg-gray-50 p-3 rounded mt-4">
                      <p className="text-sm">
                        <strong>For Healthcare Professionals:</strong> Use this tool as a decision support aid.
                        Validate findings with standard clinical protocols and consider patient-specific factors
                        not captured by the model.
                      </p>
                    </div>
                  </div>
                </DialogDescription>
              </DialogHeader>
            </DialogContent>
          </Dialog>
        </nav>
      </div>
    </header>
  );
}
