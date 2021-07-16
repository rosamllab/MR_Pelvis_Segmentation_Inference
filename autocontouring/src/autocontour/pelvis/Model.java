package autocontour.pelvis;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Model implements Closeable {
	
	static {
		System.loadLibrary("tensorflow_jni");
	}
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	
	private String inputName;
	private String outputName;
	private Session session;
	
	public Model(String modelDir, String input_name, String output_name) {
		SavedModelBundle bundle = SavedModelBundle.load(modelDir,  "serve");
		this.inputName = input_name;
		this.outputName = output_name;
		this.session = bundle.session();
	}
	
	public void close() {
		session.close();
		
	}
	
	public Tensor predict(Tensor t) {
		return session.runner().feed(inputName,  t).fetch(outputName).run().get(0);
	}
	
	public boolean[][][] predictRectum(float[][][][] meanZeroNormalized, int[] outDims, Mat[] bodyMask, int slice) {

		int yBboundary = 134;
		int xBboundary = 134;

		Mat convertedThresh = bodyMask[slice].clone();
		Mat hierarchy = new Mat();
		List<MatOfPoint> contourList = new ArrayList<MatOfPoint>();
		Imgproc.findContours(convertedThresh,  contourList,  hierarchy,  Imgproc.RETR_EXTERNAL,  Imgproc.CHAIN_APPROX_NONE);
		MatOfPoint2f[] contoursPoly = new MatOfPoint2f[contourList.size()];
		Rect[] boundRect = new Rect[contourList.size()];
		for(int i = 0; i < contourList.size(); i++) {
			contoursPoly[i] = new MatOfPoint2f();
			Imgproc.approxPolyDP(new MatOfPoint2f(contourList.get(i).toArray()), contoursPoly[i], 3, true);
			boundRect[i] = Imgproc.boundingRect(new MatOfPoint(contoursPoly[i].toArray()));

		}
		Rect bbox = new Rect();
		if (contourList.size() == 1) {
			bbox = boundRect[0];

		} else if (contourList.size() > 1) {
			int contourIndx = 0;
			int max_height = 0;
			for(int k = 0; k < contourList.size(); k++) {
				if(contourList.get(k).height() > max_height) {
					max_height = contourList.get(k).height();
					contourIndx = k;
				}
			}
			bbox = boundRect[contourIndx];
		}
		int yBoundary = bbox.y;
		int xBoundary = bbox.x;
		int height = 128;
		int width = 128;
		int bHeight = bbox.height;
		int bWidth = bbox.width;
		int yValue = yBoundary + bHeight - height;
		int xValue = 114;
		
		int img_rows = outDims[1];
		int img_columns = outDims[2];
		
		float[][][][] croppedArray = new float[1][height][width][1];
		
		for(int row = 0; row < img_rows; row++) {
			for(int column = 0; column < img_columns; column++) {
				if((row >= yValue) && (row < yValue + height) && (column >= xValue) && (column < xValue + width)) {
					croppedArray[0][row-yValue][column - xValue][0] = meanZeroNormalized[0][row][column][0];
				}
			}
		}
		
		
		Tensor tensorSlice = Tensor.create(croppedArray, Float.class);

		Tensor predResult = session.runner().feed(inputName,  tensorSlice).fetch(outputName).run().get(0);
		boolean[][][] binaryMask = new boolean[img_rows][img_columns][1];
		
		binaryMask = this.getBinaryMaskRectum(predResult, outDims, bbox);
		return binaryMask;
		
		
		
	}
	
	public boolean[][][] getBinaryMaskRectum(Tensor predResult, int[] outDims, Rect bboundary) {

		int yBoundary = bboundary.y;
		int xBoundary = bboundary.x;
		int height = 128;
		int width = 128;
		int bHeight = bboundary.height;
		int bWidth = bboundary.width;
		int yValue = yBoundary + bHeight - height;
		int xValue = 114;
	
		int img_rows = outDims[1];
		int img_columns = outDims[2];
		
		long[] resultShape = predResult.shape();
		int predRows = (int) resultShape[1];
		int predColumns = (int) resultShape[2];
		
		
		boolean[][][] binaryMask = new boolean[img_rows][img_columns][1];
		
		
		float[][][][] resultMask = new float[1][predRows][predColumns][1];
		resultMask = (float[][][][]) predResult.copyTo(new float[1][predRows][predColumns][1]);

		
		for(int row = 0; row < img_rows; row++) {
			for(int column = 0; column < img_columns; column++) {

				if ((row >= yValue) && (row < yValue + height) && (column >= xValue) && (column < xValue + width)) {
					if(resultMask[0][row - yValue][column - xValue][0] > 0.5) {
						binaryMask[row][column][0] = true;
					} else {
						binaryMask[row][column][0] = false;
					}

				} else {
					binaryMask[row][column][0] = false;
				}

			}
		}
		
		return binaryMask;
	}
	

}
