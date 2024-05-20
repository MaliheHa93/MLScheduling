package MyMLScheduling;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import MyMLScheduling.*;
import moa.core.SizeOf;

import org.cloudbus.cloudsim.Log;

import MyKnapsack.BagofPipline;
import MyKnapsack.BagofTask;
import MyKnapsack.KnapsackResult;
import net.sf.javaml.tools.weka.WekaClassifier;
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

public class KNNProvision {
	private IBk knn;
	public List<String> classValues;
	public ArrayList<Double> CostVm;
	public ArrayList<Double> scheduledWorkflowValue;
	public  int optimalK;
	public ArrayList<Attribute> attributes;
	public KNNProvision(){
		knn = new IBk();
		CostVm = new ArrayList<Double>();
		scheduledWorkflowValue = new ArrayList<Double>();
		
		attributes= new  ArrayList<Attribute>();
		attributes.add(new Attribute("bagsize"));
        attributes.add(new Attribute("maxInstanceIndex"));
        attributes.add(new Attribute("duration"));
        attributes.add(new Attribute("numTasksBag"));
        attributes.add(new Attribute("cost"));
        attributes.add(new Attribute("maxProcessTime"));
	}
	/**
	 * 
	 * @param numtaskW
	 * @param vmList
	 * @param scheduledWorkflow
	 * @param cost
	 * @param recordBOTMILP
	 * @param recordBOPMILP
	 * @param recordBOTWRPS
	 * @param recordBOPWRPS
	 * @throws Exception
	 */
		public void Train(int numtaskW, List<MyVm> vmList, Hashtable<String, Double> scheduledWorkflow, double[] cost, Hashtable<BagofTask, KnapsackResult> recordBOTMILP, Hashtable<BagofPipline, KnapsackResult> recordBOPMILP, Hashtable<BagofTask, KnapsackResult> recordBOTWRPS, Hashtable<BagofPipline, KnapsackResult> recordBOPWRPS) throws Exception {
		    ArrayList<Attribute> attributes= new ArrayList<Attribute>();
			attributes.add(new Attribute("numTask"));
			attributes.add(new Attribute("vmMips"));
			attributes.add(new Attribute("vmNumberOfPes"));
			attributes.add(new Attribute("vmRam"));
			attributes.add(new Attribute("scheduledWorkflow"));
			attributes.add(new Attribute("costVm"));
		    attributes.add(new Attribute("efficiency"));
			attributes.add(new Attribute("class"));
			
			Instances data = new Instances("Dataset", attributes, 0);
		    data.setClassIndex(data.numAttributes() - 1); // Set the class index to the last attribute

			for (Map.Entry<String, Double> entry : scheduledWorkflow.entrySet()) {
				if(entry.getKey()=="WRPS_Cost" || entry.getKey()=="MHPS_Cost") {
					CostVm.add(entry.getValue());
				}else if(entry.getKey()=="WRPS_Makespan"|| entry.getKey()=="MHPS_Makespan") {
					scheduledWorkflowValue.add(entry.getValue());
				}  
			}

			for (Map.Entry<String, Double> entry : scheduledWorkflow.entrySet()) {
				// Assuming createInstance is a method that creates an Instance object from the values
				Instance inst = createInstance(numtaskW, vmList, entry.getValue(), cost, classValues);
				data.add(inst);
			}

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
		         knn = new IBk(k);
		        
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
	    public int getOptimalK() {
		    return optimalK;
		 }
	    public void setOptimalK(int optimalK) {
		    this.optimalK=optimalK;
		 }
	    
	    /**
	     * CreateInstance function inside my train fucntion
	     * @param numtaskW
	     * @param vmList
	     * @param d
	     * @param cost
	     * @param classValues
	     * @return
	     */
		@SuppressWarnings("null")
		private Instance createInstance(int numtaskW, List<MyVm> vmList, Double d, double[] cost, List<String> classValues) {
			// TODO Auto-generated method stub
			double[] instanceValues = new double[25];
			int attributeIndex = 0; // Start from 0

			// Store numtaskW just once
			instanceValues[attributeIndex++] = numtaskW;

			// Calculate efficiency just once
			double eff = calculateEfficiency(CostVm, scheduledWorkflowValue);

			// Iterate over the remaining attributes
			for (int i = 0; i < instanceValues.length - 1; i += 4) { // Increment by 4 since you're adding 4 attributes per iteration
			    if (attributeIndex + 4 <= instanceValues.length - 1) { // Check if there's enough space in instanceValues
			        MyVm vm = vmList.get(i / 4 % vmList.size());
			        instanceValues[attributeIndex++] = vm.getMips();
			        instanceValues[attributeIndex++] = vm.getNumberOfPes();
			        instanceValues[attributeIndex++] = vm.getRam();
			        instanceValues[attributeIndex++] = d;
			        instanceValues[attributeIndex++] = cost[i / 4 % cost.length];
			    } else {
			        break; // Exit the loop if there's not enough space in instanceValues
			    }
			}
			// Store eff just once
			instanceValues[attributeIndex] = eff;
			
//			double[] instanceValues = new double[10];
//			
//			// Assuming classValues is a list of possible class values
//			double eff = calculateEfficiency(CostVm, scheduledWorkflowValue);
//			
////			int classIndex = classValues.indexOf(String.valueOf(eff));
//			
//			for (int i = 0; i < 10; i++) {
//				  instanceValues[i] = numtaskW;
//				  int vmIndex = i % vmList.size(); // Get VM index based on loop counter and vmList size (ensures recycling through VMs)
//				  MyVm vm = vmList.get(vmIndex);
//				  instanceValues[i] = vm.getMips();
//				  instanceValues[i] = vm.getNumberOfPes();
//				  instanceValues[i] = vm.getRam();
//				  instanceValues[i] = d;
//				  
//				  int CostIndex = i % cost.length;
//				  instanceValues[i] = cost[CostIndex];
//				  
//				  instanceValues[i] = eff;
//				}
//			
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
	
	/**
	 * 
	 * @param bag
	 * @param bagsize
	 * @param maxInstanceIndex
	 * @param duration
	 * @param numTasksBag
	 * @param cost
	 * @param maxProcessTime
	 * @return
	 */
	public vmProvidingResult ProvisionBoT(MyMLScheduling.BagofTask bag, int bagsize, int[] maxInstanceIndex, double duration, int[] numTasksBag, double[] cost, double[] maxProcessTime) throws Exception {
		// Create a new VMProvisioningResult
		vmProvidingResult result = new vmProvidingResult();

		// Create a new list of ResourceProvisioning
		List<ResourceProvisioning> resources = new ArrayList<ResourceProvisioning>();

		// For each task in the bag
		for (int i = 0; i < bagsize; i++) {
			// Use the KNN model to predict the resource needs for the task
			double[] features = {bagsize,maxInstanceIndex[i], duration, numTasksBag[i], cost[i], maxProcessTime[i]};
			Instance instance = createInstance(features);
			double predictedResourceNeeds = knn.classifyInstance(instance);
			// Create a new ResourceProvisioning object based on the predicted needs
			ResourceProvisioning resource = new ResourceProvisioning(i, (int) predictedResourceNeeds, numTasksBag[i], 1);
			// Add the resources list to the result
			result.resources.add(resource);
		}
        result.resources = resources;
		return result;
  }
	
	/**
	 * 
	 * @param features 
	 * @param bag
	 * @param bagsize
	 * @param maxInstanceIndex
	 * @param duration
	 * @param numTasksBag
	 * @param cost
	 * @param maxProcessTime
	 * @return
	 */
	private Instance createInstance(double[] features) {
        Instances dataset = new Instances("TaskDataset", attributes, 0);
        dataset.setClassIndex(attributes.size() - 1); // Setting the last attribute as the class
        
        // Create the instance
        Instance instance = new DenseInstance(1.0, features);
        instance.setDataset(dataset);
        return instance;
    }

/**
 * 
 * @param bop
 * @param bagsize
 * @param maxInstanceIndex
 * @param duration
 * @param numTasksPipe
 * @param cost
 * @param maxProcessTime
 * @return
 */
	public vmProvidingResult ProvisionBoP(MyMLScheduling.BagofPipline bop ,int bagsize, int[] maxInstanceIndex, double duration, int[] numTasksPipe, double[] cost, double[] maxProcessTime) throws Exception {
		vmProvidingResult result = new vmProvidingResult();
		// Create a new list of ResourceProvisioning
		List<ResourceProvisioning> resources = new ArrayList<ResourceProvisioning>();

		// For each task in the bag
		for (int i = 0; i < bagsize; i++) {
			// Use the KNN model to predict the resource needs for the task
			double[] features = {bagsize, maxInstanceIndex[i], duration, numTasksPipe[i], cost[i], maxProcessTime[i]};
			Instance instance = createInstance(features);
			double predictedResourceNeeds = knn.classifyInstance(instance);
			// Create a new ResourceProvisioning object based on the predicted needs
			ResourceProvisioning resource = new ResourceProvisioning(i, (int) predictedResourceNeeds, numTasksPipe[i], 1);
			// Add the resources list to the result
			result.resources.add(resource);
		}
		result.resources = resources;
		return result;
	}
	
	
	public void setKNN(IBk knn) {
		   this.knn = knn;
	   }
}
	
	