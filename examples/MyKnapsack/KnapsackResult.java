package MyKnapsack;

import java.util.ArrayList;
import java.util.List;

public class KnapsackResult {

	List<ResourceProvisioning> resources = new ArrayList<ResourceProvisioning>();
	double cost;
	
	public KnapsackResult getResources(ResourceProvisioning rp){
		return (KnapsackResult) resources;
		
	}
}
