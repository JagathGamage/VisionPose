import { useState } from "react";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import GraphCard from "@/components/GraphCard";  // No curly braces
import GraphModal from "@/components/GraphModal";

// In a real application, these images would be properly imported
// For this demo, we'll use placeholder images since we don't have the actual files
const graphImages = [
  { 
    id: "left",
    name: "LEFT.png", 
    title: "Left Motion Analysis",
    description: "Analysis of left-side movement patterns",
    src: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070"
  },
  { 
    id: "middle",
    name: "MIDDLE.png", 
    title: "Middle Motion Analysis",
    description: "Analysis of central movement patterns",
    src: "https://images.unsplash.com/photo-1543286386-713bdd548da4?q=80&w=2070" 
  },
  { 
    id: "right",
    name: "RIGHT.png", 
    title: "Right Motion Analysis",
    description: "Analysis of right-side movement patterns",
    src: "https://images.unsplash.com/photo-1546775392-6e1ecfd1e7d3?q=80&w=2087" 
  },
  { 
    id: "combined",
    name: "COMBINED.png", 
    title: "Combined Motion Analysis",
    description: "Comprehensive analysis of all movement patterns",
    src: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070" 
  },
];

const Index = () => {
  const [selectedGraph, setSelectedGraph] = useState<(typeof graphImages)[0] | null>(null);
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6 md:p-10">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="max-w-7xl mx-auto"
      >
        <header className="mb-12 text-center">
          <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
            Motion Analysis Dashboard
          </h1>
          <p className="text-slate-600 max-w-2xl mx-auto">
            Interactive visualization of movement patterns and analysis results across different motion scenarios
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
          {graphImages.map((graph, index) => (
            <motion.div
              key={graph.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.15 }}
            >
              <GraphCard 
                graph={graph} 
                onClick={() => setSelectedGraph(graph)} 
              />
            </motion.div>
          ))}
        </div>

        {/* Add the statistics summary card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
          className="mt-10"
        >
          <Card className="bg-white/50 backdrop-blur-sm border border-slate-200 p-6 shadow-lg">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="flex flex-col items-center p-4 bg-blue-50 rounded-lg border border-blue-100">
                <h3 className="font-semibold text-blue-800">Total Samples</h3>
                <p className="text-3xl font-bold text-blue-600">128</p>
              </div>
              <div className="flex flex-col items-center p-4 bg-purple-50 rounded-lg border border-purple-100">
                <h3 className="font-semibold text-purple-800">Accuracy</h3>
                <p className="text-3xl font-bold text-purple-600">97%</p>
              </div>
              <div className="flex flex-col items-center p-4 bg-emerald-50 rounded-lg border border-emerald-100">
                <h3 className="font-semibold text-emerald-800">Latest Update</h3>
                <p className="text-3xl font-bold text-emerald-600">Today</p>
              </div>
              <div className="flex flex-col items-center p-4 bg-amber-50 rounded-lg border border-amber-100">
                <h3 className="font-semibold text-amber-800">Analysis Status</h3>
                <p className="text-3xl font-bold text-amber-600">Complete</p>
              </div>
            </div>
          </Card>
        </motion.div>
        
        {/* Modal for viewing enlarged graphs */}
        {selectedGraph && (
          <GraphModal 
            graph={selectedGraph} 
            onClose={() => setSelectedGraph(null)} 
          />
        )}
      </motion.div>
    </div>
  );
};

export default Index;