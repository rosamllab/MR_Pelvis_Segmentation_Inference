package autocontour.pelvis;

import java.awt.Color;
import java.util.List;
import java.util.Map;


import com.mimvista.external.contouring.XMimContour;
import com.mimvista.external.control.XMimEntryPoint;
import com.mimvista.external.control.XMimSession;
import com.mimvista.external.data.XMimMutableNDArray;
import com.mimvista.external.points.XMimNoxelPointI;
import com.mimvista.external.series.XMimImage;
import com.mimvista.external.series.XMimMutableImage;

public class ContourFemrLt {
	

	private static final String pelvisContourDesc = "Automatically generates a contour for the left femoural heads";
	
	static {
		System.loadLibrary("tensorflow_jni");
	}
	

	@XMimEntryPoint(name="ContourFemrLt", author="Yasin Abdulkadir", institution="UCLA Health, Department of Radiation Oncology", category="Contouring", version="1.0", description = pelvisContourDesc)
	public static Object[] run(XMimSession sess, XMimImage img) {
		
		// Create the contour object
		XMimContour conFemrLt = img.createNewContour("O_Femr_lt");
		conFemrLt.getInfo().setColor(Color.RED);
		
		// Get the image with info
		XMimMutableImage mutable = img.getMutableCopy();
		XMimMutableNDArray array = mutable.getRawData();
		
		// Get the image dimension
		int[] dimensions = array.getDims();
		int img_rows = dimensions[1];
		int img_columns = dimensions[0];
		int img_depth = dimensions[2];
		
		// Extract image pixels and gather data for pre-processing
		float[][][][] imgPixels = new float[img_depth][img_rows][img_columns][1];
		int[] threshCounter = new int[img_depth]; // number of pixels above a threshold value (50f in this case)
		float[] sumAboveThresh = new float[img_depth]; // sum of the pixel values above the threshold
		float[] meanThresh = new float[img_depth]; // mean of the pixel values above the threshold
		float pixelValue = 0f;
		XMimNoxelPointI point = img.createNoxelPointI();
		
		for(int slice = 0; slice < img_depth; slice++) {
			threshCounter[slice] = 0;
			sumAboveThresh[slice] = 0;
			point.setCoord(2, slice);
			for(int row = 0; row < img_rows; row++) {
				point.setCoord(1,  row);
				for(int column = 0; column < img_columns; column++) {
					point.setCoord(0, column);
					pixelValue = array.getFloatValue(point);
					imgPixels[slice][row][column][0] = pixelValue;
					if (pixelValue > 50f) {
						threshCounter[slice]++;
						sumAboveThresh[slice] += pixelValue;
					}
				}
			}
			meanThresh[slice] = sumAboveThresh[slice] / threshCounter[slice];
		}
		
		// Path to the model
		String modelFemrLtPath = "C:\\Users\\yabdulkadir\\Desktop\\AutoContouring\\MIMExtensions\\MIMPelvisAutoContouring\\sampleWorkspace\\autocontouring\\resources\\model_femr_lt_10_17";
		
		// Initialize an array to store the the segmentation masks for the entire set of slices
		boolean[][][][] contourMasksFemrLt = new boolean[img_depth][img_rows][img_columns][1];
		
		// Predict segmentation masks
		contourMasksFemrLt = Utils.predictSegmentations(imgPixels, meanThresh, threshCounter, modelFemrLtPath, dimensions, contourMasksFemrLt, 80, 260);
		
		// Get all the contour volumes
		Map<String, List<Integer>> contourRanges = Utils.getContourVolumes(contourMasksFemrLt);
		
		// Get the ranges of slices with biggest contour volume
		String maxVolume = "volume 0";
		int maxVolumeRange = 0;
		int range = 0;
		for(String key : contourRanges.keySet()) {
			range = contourRanges.get(key).get(2);
			if(range > maxVolumeRange) {
				maxVolumeRange = range;
				maxVolume = key;
			}
		}
		
		int start = 0;
		int end = img_depth;
		int slicesRange = Integer.MAX_VALUE;
		if(!contourRanges.isEmpty()) {
			start = contourRanges.get(maxVolume).get(0);
			slicesRange = contourRanges.get(maxVolume).get(2);
			if(slicesRange > 69) {
				end = start + 69;
			} else {
				end = contourRanges.get(maxVolume).get(1);
			}
		}
		
		// Contour the organ
		Utils.contourOrgan(sess, img, conFemrLt, contourMasksFemrLt, start, end);
				
		return null;
		
	}
	
	
	
	
	
	
	

}
