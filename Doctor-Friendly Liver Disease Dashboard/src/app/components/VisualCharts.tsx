import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/app/components/ui/tabs';
import { BarChart3, PieChartIcon } from 'lucide-react';
import { PatientData, NORMAL_RANGES, ContributingFactor } from '@/app/types/patient';

interface VisualChartsProps {
  patientData: PatientData;
  factors: ContributingFactor[];
}

export function VisualCharts({ patientData, factors }: VisualChartsProps) {
  // Prepare data for parameter comparison chart
  const parameterData = [
    {
      name: 'Total Bilirubin',
      patient: patientData.totalBilirubin,
      normalMax: NORMAL_RANGES.totalBilirubin.max,
      unit: 'mg/dL',
    },
    {
      name: 'Direct Bilirubin',
      patient: patientData.directBilirubin,
      normalMax: NORMAL_RANGES.directBilirubin.max,
      unit: 'mg/dL',
    },
    {
      name: 'Alk Phos',
      patient: patientData.alkalinePhosphatase,
      normalMax: NORMAL_RANGES.alkalinePhosphatase.max,
      unit: 'IU/L',
    },
    {
      name: 'SGPT/ALT',
      patient: patientData.sgptAlt,
      normalMax: NORMAL_RANGES.sgptAlt.max,
      unit: 'IU/L',
    },
    {
      name: 'SGOT/AST',
      patient: patientData.sgotAst,
      normalMax: NORMAL_RANGES.sgotAst.max,
      unit: 'IU/L',
    },
    {
      name: 'Total Proteins',
      patient: patientData.totalProteins,
      normalMax: NORMAL_RANGES.totalProteins.max,
      unit: 'g/dL',
    },
    {
      name: 'Albumin',
      patient: patientData.albumin,
      normalMax: NORMAL_RANGES.albumin.max,
      unit: 'g/dL',
    },
    {
      name: 'A/G Ratio',
      patient: patientData.agRatio,
      normalMax: NORMAL_RANGES.agRatio.max,
      unit: '',
    },
  ];

  // Prepare data for contribution pie chart (only if factors exist)
  const contributionData =
    factors.length > 0
      ? factors.map((factor) => ({
          name: factor.feature,
          value: Math.round(factor.contribution * 100),
        }))
      : [];

  const COLORS = ['#dc2626', '#f59e0b', '#eab308', '#84cc16', '#22c55e'];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold">{payload[0].payload.name}</p>
          <p className="text-sm text-blue-600">
            Patient: {payload[0].value} {payload[0].payload.unit}
          </p>
          {payload[1] && (
            <p className="text-sm text-green-600">
              Normal Max: {payload[1].value} {payload[0].payload.unit}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="border-2">
      <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b">
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-purple-600" />
          Visual Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        <Tabs defaultValue="comparison" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="comparison" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Parameter Comparison
            </TabsTrigger>
            <TabsTrigger value="contribution" className="flex items-center gap-2" disabled={factors.length === 0}>
              <PieChartIcon className="w-4 h-4" />
              Contribution Weights
            </TabsTrigger>
          </TabsList>

          <TabsContent value="comparison" className="mt-6">
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-4">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-blue-500 rounded"></div>
                  <span className="text-sm text-gray-600">Patient Values</span>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <div className="w-4 h-4 bg-green-500 rounded"></div>
                  <span className="text-sm text-gray-600">Normal Upper Limit</span>
                </div>
              </div>

              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={parameterData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="name"
                    angle={-45}
                    textAnchor="end"
                    height={100}
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Bar dataKey="patient" fill="#3b82f6" name="Patient Value" />
                  <Bar dataKey="normalMax" fill="#22c55e" name="Normal Max" />
                </BarChart>
              </ResponsiveContainer>

              <p className="text-sm text-gray-600 text-center mt-4">
                Bars extending above green indicate values exceeding normal upper limits
              </p>
            </div>
          </TabsContent>

          <TabsContent value="contribution" className="mt-6">
            {factors.length > 0 ? (
              <div className="space-y-4">
                <p className="text-sm text-gray-600 mb-4">
                  Relative contribution of each factor to the overall prediction
                </p>

                <ResponsiveContainer width="100%" height={400}>
                  <PieChart>
                    <Pie
                      data={contributionData}
                      cx="50%"
                      cy="50%"
                      labelLine={true}
                      label={({ name, value }) => `${name}: ${value}%`}
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {contributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => `${value}%`} />
                  </PieChart>
                </ResponsiveContainer>

                <div className="grid grid-cols-2 gap-2 mt-4">
                  {contributionData.map((item, index) => (
                    <div key={index} className="flex items-center gap-2 text-sm">
                      <div
                        className="w-4 h-4 rounded"
                        style={{ backgroundColor: COLORS[index % COLORS.length] }}
                      ></div>
                      <span className="text-gray-700">
                        {item.name}: {item.value}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                No contributing factors to display
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
