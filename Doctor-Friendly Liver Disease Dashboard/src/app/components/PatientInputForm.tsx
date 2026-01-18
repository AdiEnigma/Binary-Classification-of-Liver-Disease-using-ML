import { useState } from 'react';
import { User, ClipboardList, TestTube, RefreshCw, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/app/components/ui/card';
import { Input } from '@/app/components/ui/input';
import { Label } from '@/app/components/ui/label';
import { Button } from '@/app/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/app/components/ui/select';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/app/components/ui/tooltip';
import { PatientData, NORMAL_RANGES } from '@/app/types/patient';

interface PatientInputFormProps {
  onPredict: (data: PatientData) => void;
  isProcessing: boolean;
}

export function PatientInputForm({ onPredict, isProcessing }: PatientInputFormProps) {
  const [formData, setFormData] = useState<PatientData>({
    patientId: '',
    name: '',
    age: 0,
    gender: '',
    totalBilirubin: 0,
    directBilirubin: 0,
    alkalinePhosphatase: 0,
    sgptAlt: 0,
    sgotAst: 0,
    totalProteins: 0,
    albumin: 0,
    agRatio: 0,
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleInputChange = (field: keyof PatientData, value: number | string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));

    // Clear error for this field
    if (errors[field]) {
      setErrors((prev) => {
        const updated = { ...prev };
        delete updated[field];
        return updated;
      });
    }
  };

  const validateForm = () => {
    const newErrors: Record<string, string> = {};

    // Age validation
    if (!formData.age || formData.age < 1 || formData.age > 120) {
      newErrors.age = 'Age must be between 1 and 120';
    }

    // Gender validation
    if (!formData.gender) {
      newErrors.gender = 'Gender is required';
    }

    // Clinical parameters validation (required + non-negative)
    const clinicalFields: Array<keyof PatientData> = [
      'totalBilirubin',
      'directBilirubin',
      'alkalinePhosphatase',
      'sgptAlt',
      'sgotAst',
      'totalProteins',
      'albumin',
      'agRatio',
    ];

    for (const field of clinicalFields) {
      const v = formData[field] as number;
      if (v === null || v === undefined || Number.isNaN(v) || v <= 0) {
        newErrors[field] = 'Required field';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) onPredict(formData);
  };

  const resetForm = () => {
    setFormData({
      patientId: '',
      name: '',
      age: 0,
      gender: '',
      totalBilirubin: 0,
      directBilirubin: 0,
      alkalinePhosphatase: 0,
      sgptAlt: 0,
      sgotAst: 0,
      totalProteins: 0,
      albumin: 0,
      agRatio: 0,
    });
    setErrors({});
  };

  const getInputClassName = (field: string, value: number) => {
    if (errors[field]) return 'border-red-500 focus:border-red-500';
    if (!value || value === 0) return '';

    const range = (NORMAL_RANGES as any)[field];
    if (range && (value < range.min || value > range.max)) {
      return 'border-amber-400 bg-amber-50';
    }
    return '';
  };

  // Remove number input spinners/arrows
  const numberInputBase =
    'appearance-none [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none';

  // Human-readable labels
  const PARAMETER_LABELS: Record<string, string> = {
    totalBilirubin: 'Total Bilirubin (TB)',
    directBilirubin: 'Direct Bilirubin (DB)',
    alkalinePhosphatase: 'Alkaline Phosphatase',
    sgptAlt: 'SGPT / ALT',
    sgotAst: 'SGOT / AST',
    totalProteins: 'Total Proteins (TP)',
    albumin: 'Albumin (ALB)',
    agRatio: 'A/G Ratio',
  };

  // Units
  const PARAMETER_UNITS: Record<string, string> = {
    totalBilirubin: 'mg/dL',
    directBilirubin: 'mg/dL',
    alkalinePhosphatase: 'IU/L',
    sgptAlt: 'IU/L',
    sgotAst: 'IU/L',
    totalProteins: 'g/dL',
    albumin: 'g/dL',
    agRatio: '',
  };

  // Field placeholders
  const PARAMETER_PLACEHOLDERS: Record<string, string> = {
    totalBilirubin: 'e.g., 0.8',
    directBilirubin: 'e.g., 0.2',
    alkalinePhosphatase: 'e.g., 85',
    sgptAlt: 'e.g., 28',
    sgotAst: 'e.g., 22',
    totalProteins: 'e.g., 7.2',
    albumin: 'e.g., 4.5',
    agRatio: 'e.g., 1.67',
  };

  // Step values
  const PARAMETER_STEPS: Record<string, number> = {
    totalBilirubin: 0.1,
    directBilirubin: 0.1,
    alkalinePhosphatase: 1,
    sgptAlt: 1,
    sgotAst: 1,
    totalProteins: 0.1,
    albumin: 0.1,
    agRatio: 0.01,
  };

  return (
    <Card className="h-full">
      <CardHeader className="bg-gradient-to-r from-teal-50 to-cyan-50 border-b">
        <CardTitle className="flex items-center gap-2 text-xl">
          <ClipboardList className="w-5 h-5 text-teal-600" />
          Patient Information & Lab Results
        </CardTitle>
        <CardDescription>Enter patient details and clinical laboratory values</CardDescription>
      </CardHeader>

      <CardContent className="p-6 overflow-y-auto max-h-[calc(100vh-240px)]">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Patient Details */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b">
              <User className="w-4 h-4 text-gray-600" />
              <h3 className="font-semibold text-gray-900">Patient Details</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Patient ID */}
              <div className="space-y-2">
                <Label htmlFor="patientId">Patient ID</Label>
                <Input
                  id="patientId"
                  placeholder="e.g., LP-1234"
                  value={formData.patientId}
                  onChange={(e) => handleInputChange('patientId', e.target.value)}
                />
              </div>

              {/* Patient Name */}
              <div className="space-y-2">
                <Label htmlFor="name">Patient Name</Label>
                <Input
                  id="name"
                  placeholder="Enter patient name"
                  value={formData.name}
                  onChange={(e) => handleInputChange('name', e.target.value)}
                />
              </div>

              {/* Age */}
              <div className="space-y-2">
                <Label htmlFor="age" className="flex items-center gap-1">
                  Age <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="age"
                  type="number"
                  min={1}
                  max={120}
                  placeholder="Enter age"
                  value={formData.age || ''}
                  onChange={(e) => handleInputChange('age', parseInt(e.target.value) || 0)}
                  className={`${numberInputBase} ${errors.age ? 'border-red-500' : ''}`}
                />
                {errors.age && <p className="text-xs text-red-500">{errors.age}</p>}
              </div>

              {/* Gender */}
              <div className="space-y-2">
                <Label htmlFor="gender" className="flex items-center gap-1">
                  Gender <span className="text-red-500">*</span>
                </Label>
                <Select
                  value={formData.gender}
                  onValueChange={(value) => handleInputChange('gender', value as 'Male' | 'Female')}
                >
                  <SelectTrigger className={errors.gender ? 'border-red-500' : ''}>
                    <SelectValue placeholder="Select gender" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Male">Male</SelectItem>
                    <SelectItem value="Female">Female</SelectItem>
                  </SelectContent>
                </Select>
                {errors.gender && <p className="text-xs text-red-500">{errors.gender}</p>}
              </div>
            </div>
          </div>

          {/* Clinical Parameters */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b">
              <TestTube className="w-4 h-4 text-gray-600" />
              <h3 className="font-semibold text-gray-900">Clinical Laboratory Parameters</h3>
              <span className="text-red-500 text-sm">* Required</span>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-blue-800">
                Values outside normal ranges will be highlighted. Hover over field labels to see normal ranges.
              </p>
            </div>

            <TooltipProvider>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(NORMAL_RANGES).map(([key, range]) => (
                  <div key={key} className="space-y-2">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Label className="flex items-center gap-1 cursor-help">
                          {PARAMETER_LABELS[key] ?? key}
                          <span className="text-red-500">*</span>
                        </Label>
                      </TooltipTrigger>

                      <TooltipContent>
                        <p>{(range as any).description}</p>
                      </TooltipContent>
                    </Tooltip>

                    <div className="relative">
                      <Input
                        type="number"
                        step={PARAMETER_STEPS[key] ?? 0.01}
                        min={(range as any).min}
                        max={(range as any).max}
                        placeholder={PARAMETER_PLACEHOLDERS[key] ?? ''}
                        value={(formData as any)[key] || ''}
                        onChange={(e) =>
                          handleInputChange(key as keyof PatientData, parseFloat(e.target.value) || 0)
                        }
                        className={`${numberInputBase} ${getInputClassName(
                          key,
                          (formData as any)[key]
                        )}`}
                      />

                      {!!PARAMETER_UNITS[key] && (
                        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500">
                          {PARAMETER_UNITS[key]}
                        </span>
                      )}
                    </div>

                    {errors[key] && <p className="text-xs text-red-500">{errors[key]}</p>}
                  </div>
                ))}
              </div>
            </TooltipProvider>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-3 pt-4 border-t">
            <Button
              type="submit"
              className="bg-teal-600 hover:bg-teal-700 flex-1 min-w-[200px]"
              disabled={isProcessing}
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <TestTube className="w-4 h-4 mr-2" />
                  Predict Risk
                </>
              )}
            </Button>

            <Button
              type="button"
              variant="ghost"
              onClick={resetForm}
              disabled={isProcessing}
              className="min-w-[120px]"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Reset
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
