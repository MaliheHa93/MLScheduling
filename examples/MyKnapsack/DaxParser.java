package MyKnapsack;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.DoubleToLongFunction;

import org.cloudbus.cloudsim.Log;
import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;
import MyKnapsack.MyTask;
import MyKnapsack.MyConstant;



public class DaxParser {
	
	private static ArrayList<MyTask> taskList;
	
	private static Hashtable<Integer,ArrayList<MyTask>> taskLevel;
	
	private static Map<String , MyTask> mNameTask;
	
	private static int taskId=0;
	
	private int level;
	
	public static  int[][] levelTaskNumber;
	
	public final static int FILE_TYPE_INPUT = 1;
	public final static int FILE_TYPE_OUTPUT = 2;
	
	public static ArrayList<Integer> depth;
	
	
	public static Workflow ParseDax(String daxPath){	
		
		try {
			taskList =  new ArrayList<>();
			mNameTask = new Hashtable<>();
			depth = new ArrayList<>();
			Random random = new Random();
			
			SAXBuilder dp = new SAXBuilder();
			Document d = dp.build(new File(daxPath));
			Element root = d.getRootElement();
			List<Element> list = root.getChildren();
			double rangeMin = -1.0;
			double rangeMax  = 1.0;
			
			for(Element node: list) {
					switch(node.getName().toLowerCase()) {
						case "job":
							long length;
							String nodeId = node.getAttributeValue("id");
							String nodeType = node.getAttributeValue("name");
							String runtimeNode = node.getAttributeValue("runtime");							
							
							double runTime = Double.parseDouble(runtimeNode);
//							runTime *= VMOffersGoogle.getMips(VMOffersGoogle.n1_standard_1);
//							runTime /= VMOffersGoogle.getMips(VMOffersGoogle.n1_standard_8);
//							
							if(runTime <= 0) {
								runTime = 5;
							}
							
//							runTime += ((random.nextDouble() * 0.2) - 0.1);
//							runTime = Math.max(0, runTime);
							length = (long) (runTime * VMOffersGoogle.getMips(VMOffersGoogle.n1_standard_8));
							
							/*get all file of each task*/
							List<Element> fileList = node.getChildren();
		                    ArrayList<org.cloudbus.cloudsim.File> mFileList = new ArrayList<org.cloudbus.cloudsim.File>();
		                    for(Element element: fileList) {
		                    	if(element.getName().toLowerCase() == "uses") {
		                    		String link =  element.getAttributeValue("link");
		                    		String fileName = element.getAttributeValue("file");
		                    		
		                    		long sizeFile = Long.parseLong(element.getAttributeValue("size"));
		                    		long sizeinMb = (long) Math.ceil(sizeFile / (1024 * 1024));
		                    		/**
	                                 * a bug of cloudsim, size 0 causes a problem. 1
	                                 * is ok.
	                                 */
	                                if (sizeinMb == 0) {
	                                	sizeinMb++;
	                                }
	                                int type = 0;
	                                switch(link) {
	                                	case "input" :
	                                		type = FILE_TYPE_INPUT;	                                		
	                                		break;
	                                	case "output":
	                                		type = FILE_TYPE_OUTPUT;	                                		
	                                		break;
	                                	default : 
	                                		Log.printLine("Parsing Error");
	                                		break;
	                                }
	                                org.cloudbus.cloudsim.File tFile;
	                                /*
	                                 * Already exists an input file (forget output file)
	                                 */
	                                if (sizeFile < 0) {
	                                    /*
	                                     * Assuming it is a parsing error
	                                     */
	                                	sizeFile = 0 - sizeFile;
//	                                    Log.printLine("Size is negative, I assume it is a parser error");
	                                }
	                                tFile = new org.cloudbus.cloudsim.File(fileName, type);
	                                tFile.setType(type);
	                                tFile.setFileSize(sizeinMb);
	                                mFileList.add(tFile);
		                    	}
		                    }
		                    MyTask task = new MyTask(taskId++, length , runTime , nodeId ,nodeType);
		                    mNameTask.put(nodeId,task);
		                    
		                    for (Iterator<org.cloudbus.cloudsim.File> it = mFileList.iterator();it.hasNext();) {
		                    	org.cloudbus.cloudsim.File file = (org.cloudbus.cloudsim.File) it.next();
								task.addRequiredFile(file.getName());
								if(file.getType() == FILE_TYPE_INPUT) {
									task.addDataDependencies(file);
								}
								else {
									task.addOutput(file);;
								}
							}
		                    task.setFileList(mFileList);
		                    taskList.add(task);
		                    break;
		                    
						case "child":
							List<Element> parentlist = node.getChildren();
							String name = node.getAttributeValue("ref");
							if(mNameTask.containsKey(name)) {
								MyTask childtask = (MyTask) mNameTask.get(name);
								for(Element p:parentlist) {
									String namep = p.getAttributeValue("ref");
									if(mNameTask.containsKey(namep)) {
										MyTask pt = (MyTask) mNameTask.get(namep);
										pt.addChild(childtask);
										childtask.addParent(pt);
									}
									
								}
							}								
						}		
					}
			  /**
             * If a task has no parent, then it is root task.
             */
            ArrayList<Object> roots = new ArrayList<>();
            for (MyTask task : mNameTask.values()) {
                task.setDepth(0);
                if (task.getParentList().isEmpty()) {
                    roots.add(task);
                }
            }
            /**
             * Add depth from top to bottom.
             */
            for (Iterator it = roots.iterator(); it.hasNext();) {
                MyTask task = (MyTask) it.next();
                setDepth(task, 1);
            }
            
            /**
             * Clean them so as to save memory. Parsing workflow may take much
             * memory
             */
            mNameTask.clear();
            
            return new Workflow(taskList);
			
		}catch (JDOMException jde) {
            Log.printLine("JDOM Exception;Please make sure your dax file is valid");

        } catch (IOException ioe) {
            Log.printLine("IO Exception;Please make sure dax.path is correctly set in your config file");

        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("Parsing Exception");
        }
		return null;
	}

	private static void setDepth(MyTask task, int depth) {
		if (depth > task.getDepth()) {
			task.setDepth(depth);
		}
		for (MyTask cTask : task.getChildList()) {
			setDepth(cTask, task.getDepth() + 1);
		}
		
	}
	public ArrayList<MyTask> getTaskList() {
		return taskList;
		
	}

}
