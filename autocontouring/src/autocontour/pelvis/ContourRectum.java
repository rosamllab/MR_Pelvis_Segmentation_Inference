package autocontour.pelvis;

import java.awt.Color;
import java.util.Collection;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import com.mimvista.external.contouring.XMimContour;
import com.mimvista.external.control.XMimEntryPoint;
import com.mimvista.external.control.XMimSession;
import com.mimvista.external.data.XMimMutableNDArray;
import com.mimvista.external.linking.XMimLinkController;
import com.mimvista.external.points.XMimNoxelPointF;
import com.mimvista.external.points.XMimNoxelPointI;
import com.mimvista.external.series.XMimImage;
import com.mimvista.external.series.XMimMutableImage;

public class ContourRectum {
	

	private static final String pelvisContourDesc = "Automatically generates a contour for the rectum";
	
	static {
		System.loadLibrary("tensorflow_jni");
	}
	
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	
	
	@XMimEntryPoint(name="ContourRectum", author="Yasin Abdulkadir", institution="UCLA Health, Department of Radiation Oncology", category="Contouring", version="1.0", description = pelvisContourDesc)
	public static Object[] run(XMimSession sess, XMimImage img) {
		
		// Create the contour object
		XMimContour conRectum = img.createNewContour("O_Rctm");
		conRectum.getInfo().setColor(new Color(191, 146, 96));
		
		// Get the image with info
		XMimMutableImage mutable = img.getMutableCopy();
		XMimMutableNDArray array = mutable.getRawData();
		
		// Get the image dimensions
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
		String modelRectumPath = "C:\\Users\\yabdulkadir\\Desktop\\AutoContouring\\MIMExtensions\\MIMPelvisAutoContouring\\sampleWorkspace\\autocontouring\\resources\\model_rectum_11_16";
		
		// Initialize an array to store the the segmentation masks for the entire set of slices
		boolean[][][][] contourMasksRectum = new boolean[img_depth][img_rows][img_columns][1];
		
		// Determine the starting and ending slices of right femur contours 
		int startSlice = 0;
		int endSlice = img_depth;

		Collection<XMimContour> allContours = img.getContours();
		boolean foundBodyContour = false;
		boolean foundFemrRtContour = false;

		for (XMimContour con : allContours) {
			int[] dims;
			
			if(con.getInfo().getName().equals("O_Femr_rt")) {
				foundFemrRtContour = true;
				dims = con.getDims();
				XMimNoxelPointI contourNoxelF = sess.getPointFactory().createNoxelPointI(
						con.getSpace(),
						new int[con.getDims().length]);

				// another for the image
				XMimNoxelPointF imageNoxel = img.createNoxelPointF();

				// a linker to convert from contour space to the image space
				XMimLinkController linker = sess.getLinker();
				boolean sliceStarted = false;
				int countToStart = 0;
				
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
								if((con.getData().getBooleanValue(contourNoxelF) == true) && ( sliceStarted == false)) {
									countToStart++;
									if(countToStart == 5) {
										sliceStarted = true;
										startSlice = depth;
									}
								} else if((con.getData().getBooleanValue(contourNoxelF) == true) && (sliceStarted == true)) {
									endSlice = depth;
								}
								
								
							} catch (Exception e) {
								//exception = e.toString();
								break;
							}
						}
					}
				}
			}
			
		}
		
		// Get the body contour
		Mat[] bodyMask = new Mat[img_depth];
		for (XMimContour con : allContours) {
			int[] dims;
			if(con.getInfo().getName().equals("Body")) {
				foundBodyContour = true;
				dims = con.getDims();
				XMimNoxelPointI contourNoxelF = sess.getPointFactory().createNoxelPointI(
						con.getSpace(),
						new int[con.getDims().length]);

				// another for the image
				XMimNoxelPointF imageNoxel = img.createNoxelPointF();

				// a linker to convert from contour space to the image space
				XMimLinkController linker = sess.getLinker();
				for(int z = 0; z < dims[2]; z++) {
					contourNoxelF.setCoord(2,  z);
					
					bodyMask[z] = new Mat(img_rows, img_columns, CvType.CV_8U);
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
								if((con.getData().getBooleanValue(contourNoxelF) == true)) {
									bodyMask[depth].put(row, column, 255);
								} else {
									bodyMask[depth].put(row,  column, 0);
								}
							} catch (Exception e) {
								e.toString();
							}
						}
					}
				}
			}
				
		}
		
		
		

		// Contour the organ
		try {
			if (foundFemrRtContour == true) {
				if (foundBodyContour == true) {
					contourMasksRectum = Utils.predictRectumSegmentations(imgPixels, 
							meanThresh, 
							threshCounter, 
							modelRectumPath, 
							dimensions, 
							contourMasksRectum, 
							bodyMask, 
							startSlice, 
							endSlice);
					Utils.contourOrgan(sess, 
							img, 
							conRectum, 
							contourMasksRectum, 
							startSlice, 
							endSlice);
				} 
			} 
			
		} catch (Exception e) {
			e.toString();
		}
				
		return null;
		
	}
	
	
	

}
