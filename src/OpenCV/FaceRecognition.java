package OpenCV;


import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import com.google.gson.JsonObject;

import javafx.application.*;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

public class FaceRecognition {
	
	private String ImagePath;
	
	

	
	public int FaceRecognitionProcess(String ImagePath) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		Mat face = Imgcodecs.imread(ImagePath, Imgcodecs.IMREAD_GRAYSCALE);
		
		FaceDetectionM FC = new FaceDetectionM();
		ArrayList<JsonObject> json = FC.ReadJsonArray("Images.json");
		ArrayList<Mat> faces = new ArrayList<Mat>();
		for (int i = 0; i < json.size(); i++) {
			faces.add(FaceDetectionM.matFromJson(json.get(i)));
		}
		Labels labels = new Labels();
		RecognizeFace RF = new RecognizeFace(faces, labels.readingInteger());
		int Id = (int) RF.RecognizeFaceAlgorithm(face);
		return Id;
	}
	

}
