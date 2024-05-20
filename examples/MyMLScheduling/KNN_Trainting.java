package MyMLScheduling;

//import MyMLScheduling.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import MyKnapsack.BagofPipline;
import MyKnapsack.BagofTask;
import MyKnapsack.KnapsackResult;

import java.lang.Double;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.functions.SGD;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;

public final class KNN_Trainting {
	// train KNN with workflow data including runtime of tasks, vmlists,
	public ArrayList<Attribute> attributes;
	public List<String> classValues;
	public ArrayList<Double> CostVm;
	public ArrayList<Double> scheduledWorkflowValue;
	public  int optimalK;
    public  IBk knn;

	public KNN_Trainting() {
		knn = new IBk();
		attributes= new  ArrayList<Attribute>();
		attributes.add(new Attribute("numTask"));
		attributes.add(new Attribute("vmList"));
		attributes.add(new Attribute("scheduledWorkflow"));
		attributes.add(new Attribute("BOT"));
		attributes.add(new Attribute("BOP"));
		attributes.add(new Attribute("costVm"));
		attributes.add(new Attribute("class"));

		CostVm = new ArrayList<Double>();
		scheduledWorkflowValue = new ArrayList<Double>();
		classValues = Arrays.asList("Efficient", "Moderately Efficient", "Inefficient");
  }
/**
 * 
 * @param numtaskW
 * @param vmList
 * @param scheduledWorkflow
 * @param costVm
 * @param recordBOTMILP
 * @param recordBOPMILP
 * @param recordBOTWRPS
 * @param recordBOPWRPS
 * @throws Exception
 */
	public void Train(int numtaskW, List<MyVm> vmList, Hashtable<String, Double> scheduledWorkflow, ArrayList<Double> costVm, Hashtable<BagofTask, KnapsackResult> recordBOTMILP, Hashtable<BagofPipline, KnapsackResult> recordBOPMILP, Hashtable<BagofTask, KnapsackResult> recordBOTWRPS, Hashtable<BagofPipline, KnapsackResult> recordBOPWRPS) throws Exception {

		Instances data = new Instances("Dataset", attributes, 0);
		for (Map.Entry<String, Double> entry : scheduledWorkflow.entrySet()) {
			if(entry.getKey()=="WRPS_Cost" || entry.getKey()=="MHPS_Cost") {
				CostVm.add(entry.getValue());
			}else if(entry.getKey()=="WRPS_Makespan"|| entry.getKey()=="MHPS_Makespan") {
				scheduledWorkflowValue.add(entry.getValue());
			}  
		}

		for (Map.Entry<String, Double> entry : scheduledWorkflow.entrySet()) {
			// Assuming createInstance is a method that creates an Instance object from the values
			Instance inst = createInstance(numtaskW, vmList, entry.getValue(), costVm, classValues);
			data.add(inst);
		}

		data.setClassIndex(data.numAttributes() - 1);
		//		calculating optimal K for K nearest algorithm
		int[] ks = {5, 10, 20, 50, 100, 150, 200};
		double bestAccuracy = 0;
		int bestK = ks[0];

		for (int k : ks) {
			double accuracy = crossValidateK(data, k);
			if (accuracy > bestAccuracy) {
				bestAccuracy = accuracy;
				bestK = k;
			}
		}
		this.optimalK = bestK;
		this.setOptimalK(bestK);
		knn = new IBk(optimalK);
		knn.buildClassifier(data);
	}
/**
 * 
 * @param data
 * @param k
 * @return
 * @throws Exception
 */
	 private double crossValidateK(Instances data, int k) throws Exception {
	        IBk knn = new IBk(k);
	        
	        int numFolds = Math.min(10, data.numInstances());
	        double accuracy = 0;
	       

	        for (int fold = 0; fold < numFolds;) {
	            Instances train = data.trainCV(numFolds, fold);
	            Instances test = data.testCV(numFolds, fold);

	            knn.buildClassifier(train);
	            int correct = 0;

	            for (int i = 0; i < test.numInstances()-1; i++) {
	                double predicted = knn.classifyInstance(test.instance(i));
	                if (predicted == test.instance(i).classValue()) {
	                    correct++;
	                }
	            }
	            accuracy += (double) correct / test.numInstances();
	         // Break out of the loop after calculating fold accuracy
	            break;
	        }
	        return accuracy / numFolds;
	    }

   public IBk getKNN() {
	   return knn;
	}
   public void setKNN(IBk knn) {
	   this.knn=knn;
   }
    public int getOptimalK() {
	    return optimalK;
	 }
    public void setOptimalK(int optimalK) {
	    this.optimalK=optimalK;
	 }
    
    /**
     * 
     * @param numtaskW
     * @param vmList
     * @param d
     * @param costVm
     * @param classValues
     * @return
     */
	@SuppressWarnings("null")
	private Instance createInstance(int numtaskW, List<MyVm> vmList, Double d, ArrayList<Double> costVm, List<String> classValues) {
		// TODO Auto-generated method stub
		double[] instanceValues = new double[10];
		
		// Assuming classValues is a list of possible class values
		double eff = calculateEfficiency(CostVm, scheduledWorkflowValue);
		
//		int classIndex = classValues.indexOf(String.valueOf(eff));
		
		for (int i = 0; i < 10; i++) {
			  instanceValues[i] = numtaskW;
			  int vmIndex = i % vmList.size(); // Get VM index based on loop counter and vmList size (ensures recycling through VMs)
			  MyVm vm = vmList.get(vmIndex);
			  instanceValues[i] = vm.getMips();
			  instanceValues[i] = vm.getNumberOfPes();
			  instanceValues[i] = vm.getRam();
			  instanceValues[i] = d;
			  instanceValues[i] = costVm.get(0);
			  instanceValues[i] = eff;
			}
		
		 return new DenseInstance(1.0, instanceValues);
	}
	/**
	 * 
	 * @param costVm2
	 * @param scheduledWorkflowValue2
	 * @return
	 */
	private double calculateEfficiency(ArrayList<Double> costVm2, ArrayList<Double> scheduledWorkflowValue2) {
		
		double w1=0.7, w2 = 0.3;
		
	    double min_cost = getMinimumCost(costVm2);
		double max_cost = getMaximumCost(costVm2);
		double normalized_cost;
		double normalized_makespan;
		double min_makespan = getMinimumMakespan(scheduledWorkflowValue2);
		double max_makespan = getMaximumMakespan(scheduledWorkflowValue2);
		double epsilon = 0.001; // Small epsilon value to avoid division by zero

		if (costVm2 == null) {
			  // Handle case where cost is missing (assign a default value or throw an exception)
			  throw new RuntimeException("Cost value missing in hashtable");
		}
		
		 normalized_cost = (costVm2.get(1) - min_cost) / (max_cost - min_cost + epsilon);
		if (min_cost == max_cost) {
		    // Assign a small non-zero value to avoid division by zero (adjust as needed)
		    normalized_cost = 0.001;
		  }
		
		  normalized_makespan = (scheduledWorkflowValue2.get(1) - min_makespan) / (max_makespan - min_makespan + epsilon);
		  // Handle potential division by zero (check if min_makespan == max_makespan)
		 if (min_makespan == max_makespan) {
		    // Assign a small non-zero value to avoid division by zero (adjust as needed)
		    normalized_makespan = 0.001;
		  }
		double efficiency = -w1 * (normalized_cost) + w2 * normalized_makespan;

	   return efficiency;
		
	}
	/**
	 * 
	 * @param scheduledWorkflowValue2
	 * @return
	 */
	private double getMaximumMakespan(ArrayList<Double> scheduledWorkflowValue2) {
		// TODO Auto-generated method stub
		double maxtime = Double.MIN_VALUE;
		for(Double exectime:scheduledWorkflowValue2) {
			if(maxtime < exectime) {
				maxtime = exectime;
			}
		}
		return maxtime;
	}
	/**
	 * 
	 * @param scheduledWorkflowValue2
	 * @return
	 */
	private double getMinimumMakespan(ArrayList<Double> scheduledWorkflowValue2) {
		// TODO Auto-generated method stub
		double minTime = Double.MAX_VALUE;
		for(Double exectime:scheduledWorkflowValue2) {
			if(minTime > exectime) {
				minTime=exectime;
			}
		}
		return minTime;
	}
	/**
	 * 
	 * @param costVm2
	 * @return
	 */
	private double getMaximumCost(ArrayList<Double> costVm2) {
		double maxcost= Double.MIN_VALUE;
		for(Double execcost:costVm2) {
			if(maxcost < execcost) {
				maxcost= execcost;
			}
		}
		return maxcost;
	}
	/**
	 * 
	 * @param costVm2
	 * @return
	 */
	private double getMinimumCost(ArrayList<Double> costVm2) {
		// TODO Auto-generated method stub
		double mincost= Double.MAX_VALUE;
		for(Double execcost:costVm2) {
			if(mincost > execcost) {
				mincost= execcost;
			}
		}
		return mincost;
	}
}