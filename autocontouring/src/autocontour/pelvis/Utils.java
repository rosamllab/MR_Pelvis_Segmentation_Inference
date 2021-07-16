package autocontour.pelvis;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Mat;
import org.tensorflow.Tensor;

import com.mimvista.external.contouring.XMimContour;
import com.mimvista.external.control.XMimSession;
import com.mimvista.external.linking.XMimLinkController;
import com.mimvista.external.points.XMimNoxelPointF;
import com.mimvista.external.points.XMimNoxelPointI;
import com.mimvista.external.series.XMimImage;

public class Utils {

	public Utils() {
		
	}
	
	public static float[][][][] meanZeroNormalize(float[][][] sliceArray, float sliceMean, int numAboveThresh) {
		// Pre-process each slice of the image by subtracting the mean of the slice above a specified threshold and dividing it by the standard deviation
		// then truncate the values that are above and below 5f and -5f followed by normalizing the values to have values between 0 and 1

		float theDiff = 0;
		float variance = 0;
		int img_rows = sliceArray.length;
		int img_columns = sliceArray[0].length;

		for(int row = 0; row < img_rows; row++) {
			for(int column = 0; column < img_columns; column++) {
				if(sliceArray[row][column][0] > 50f) {
					theDiff = sliceArray[row][column][0] - sliceMean;
					variance += (theDiff * theDiff) / (float) numAboveThresh;
				}
			}
		}

		float stdSlice = (float) Math.sqrt(variance);
		float theMin = Float.MAX_VALUE;
		float theMax = Float.MIN_VALUE;
		float pixelValue;


		for(int row = 0; row < img_rows; row++) {
			for(int column = 0; column < img_columns; column++) {
				pixelValue = sliceArray[row][column][0];
				pixelValue = (pixelValue - sliceMean) / stdSlice;
				if(pixelValue > 5f) {
					pixelValue = 5f;
					sliceArray[row][column][0] = pixelValue;

				} else if(pixelValue < -5f) {
					pixelValue = -5f;
					sliceArray[row][column][0] = pixelValue;
				} else {
					sliceArray[row][column][0] = pixelValue;
				}

				if(pixelValue > theMax) {
					theMax = pixelValue;
				} else if(pixelValue < theMin) {
					theMin = pixelValue;
				}

			}
		}

		theMax = theMax + (float) Math.abs(theMin);

		for(int row = 0; row < img_rows; row++) {
			for(int column = 0; column < img_columns; column++) {
				pixelValue = sliceArray[row][column][0];
				pixelValue = pixelValue + (float) Math.abs(theMin);
				pixelValue = pixelValue / theMax;
				sliceArray[row][column][0] = pixelValue;			
			}
		}

		float[][][][] normalizedSlice = new float[1][img_rows][img_columns][1];

		normalizedSlice[0] = sliceArray;

		return normalizedSlice;
	}
	
	public static boolean[][][] getBinaryMask(Tensor predResult) {
		// Get a binary mask from the tensorflow prediction results by a simple thresholding above 0.5
		
		long[] resultShape = predResult.shape();
		int img_rows = (int) resultShape[1];
		int img_columns = (int) resultShape[2];
		boolean[][][] binaryMask = new boolean[img_rows][img_columns][1];
		
		
		float[][][][] resultMask = new float[1][img_rows][img_columns][1];
		resultMask = (float[][][][]) predResult.copyTo(new float[1][img_rows][img_columns][1]);
		
		for(int row = 0; row < img_rows; row++) {
			for(int column = 0; column < img_columns; column++) {
				if(resultMask[0][row][column][0] > 0.5) {
					binaryMask[row][column][0] = true;
				} else {
					binaryMask[row][column][0] = false;
				}
			}
		}
		
		return binaryMask;
	}
	
	public static boolean[][][][] predictSegmentations(float[][][][] imgPixels, float[] meanThresh, int[] threshCounter, String modelPath, int[] dimensions, boolean[][][][] contourMasks, int startSlice, int endSlice) {
		
		// Run the tensorflow model to predict the segmentation mask
		
		final String INPUTNAME = "serving_default_input_1";
		final String OUTPUTNAME = "StatefulPartitionedCall";
		
		try (Model model = new Model(modelPath, INPUTNAME, OUTPUTNAME)) {
			int imgRows = dimensions[1];
			int imgColumns = dimensions[0];
			int imgDepth = dimensions[2];

			float[][][][] meanZeroNormalized = new float[1][imgRows][imgColumns][1];


			for(int slice = startSlice; slice < endSlice; slice++) {
				meanZeroNormalized = meanZeroNormalize(imgPixels[slice], meanThresh[slice], threshCounter[slice]);


				Tensor tensorSlice = Tensor.create(meanZeroNormalized, Float.class);

				Tensor result = model.predict(tensorSlice);
				contourMasks[slice] = getBinaryMask(result);
				result.close();
				tensorSlice.close();



			}

		}
		
		return contourMasks;
	}
	public static boolean[][][][] predictRectumSegmentations(float[][][][] imgPixels, float[] meanThresh, int[] threshCounter, String modelPath, int[] dimensions, boolean[][][][] contourMasks, Mat[] bodyMask, int startSlice, int endSlice) {
		
		
		final String INPUTNAME = "serving_default_input_1";
		final String OUTPUTNAME = "StatefulPartitionedCall";

		try (Model model = new Model(modelPath, INPUTNAME, OUTPUTNAME)) {
			int imgRows = dimensions[1];
			int imgColumns = dimensions[0];
			int imgDepth = dimensions[2];

			float[][][][] meanZeroNormalized = new float[1][imgRows][imgColumns][1];


			for(int slice = startSlice; slice <= endSlice; slice++) {
				meanZeroNormalized = meanZeroNormalize(imgPixels[slice], meanThresh[slice], threshCounter[slice]);

				
				contourMasks[slice] = model.predictRectum(meanZeroNormalized, dimensions, bodyMask, slice);

			}

		}

		return contourMasks;
	}
	
	
	public static void contourOrgan(XMimSession sess, XMimImage img, XMimContour contour, boolean[][][][] contourMasks,int start,int end) {
		// create a point at the multiplied voxel zero for the contour
		XMimNoxelPointI contourNoxelF = sess.getPointFactory().createNoxelPointI(
				contour.getSpace(),
				new int[contour.getDims().length]);

		// another for the image
		XMimNoxelPointF imageNoxel = img.createNoxelPointF();

		// a linker to convert from contour space to the image space
		XMimLinkController linker = sess.getLinker();

		// for the exception stack trace
		String exception = "";
		
		// contour dimensions
		int[] dims = contour.getDims();
		
		// draw the contour
		int[] indicesLt = {0,0,0};
		for(int z = 0; z < dims[2]; z++) {
			contourNoxelF.setCoord(2,  z);

			for(int y = 0; y < dims[1]; y++) {
				contourNoxelF.setCoord(1, y);

				for(int x = 0; x < dims[0]; x++) {
					contourNoxelF.setCoord(0,  x);
					// we have to convert from contour space to image space
					linker.convertPoint(
							contourNoxelF.toVoxelCenter(),
							img.getSpace(),
							null,
							imageNoxel);
					try {

						int depth = (int) Math.round(Math.floor(imageNoxel.toArray()[2]));
						int row = (int) Math.round(Math.floor(imageNoxel.toArray()[1]));
						int column = (int) Math.round(Math.floor(imageNoxel.toArray()[0]));
						if((depth >= start) && (depth <= end)) {
							contour.getData().setRange(contourNoxelF,  (boolean) contourMasks[depth][row][column][0], 1);
						}
					} catch (Exception e) {
						exception = e.toString();
						break;
					}
				}
			}
		}

		contour.redrawCompletely();
		contour.sanitize();
	}
	
	public static Map<String, List<Integer>> getContourVolumes(boolean[][][][] contourMasks) {
		// Get all the disconnected contour volumes
		
		Map<String, List<Integer>> contourRanges = new HashMap<>();
		int contourDepth = contourMasks.length;
		int contourRows = contourMasks[0].length;
		int contourColumns = contourMasks[0][0].length;
		boolean doneWithSlice;
		int counter;
		int start = 0;
		int end = 0;
		int range = 0;
		boolean started = false;
		int volume = 0;
		String contourVolume;
		for(int slice = 0; slice < contourDepth; slice++) {
			doneWithSlice = false;
			counter = 0;
			for(int row = 0; row < contourRows; row++) {
				
				for(int column = 0; column < contourColumns; column++) {
					
					if(doneWithSlice) {
						continue;
					} else {
						if(counter == 5) {
							doneWithSlice = true;
							if(!started) {
								started = true;
								start = slice;
							}
						} else {
							if(contourMasks[slice][row][column][0]) {
								counter++;
							}
						}
					}
				}
			}
			
			if(started) {
				if(counter < 5) {
					end = slice;
					started = false;
					range = end - start;
					volume++;
					contourVolume = "volume " + String.format("%d", volume);
					List<Integer> volumeInfo = new ArrayList<>();
					volumeInfo.add(start);
					volumeInfo.add(end);
					volumeInfo.add(range);
					contourRanges.put(contourVolume, volumeInfo);
				}
			}
		}
		
		return contourRanges;
	}
}
