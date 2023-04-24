package yoloxray;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.utils.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.dnn.DetectionModel;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class Yolo {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws InterruptedException, IOException {
        Scanner selection = new Scanner(System.in);
        boolean isTrue = false;
        while (isTrue == false) {
            System.out.print("Welecome! Enter 'video' to proccess video or Enter 'image' to process image:");
            isTrue = choice(selection.nextLine());

        }
    }

    public static boolean choice(String str) throws InterruptedException, IOException {
        if (str.equalsIgnoreCase("video")) {
            video();
            return true;
        }
        if (str.equalsIgnoreCase("image")) {
            image();
            return true;
        }
        System.out.println("Invalid");
        return false;

    }

    private static List<String> getOutputNames(Net net) {

        List<String> names = new ArrayList<>();
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layerNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layerNames.get(item - 1)));
        return names;
    }

    public static void video() throws InterruptedException, IOException {

        Scanner path = new Scanner(System.in);
        System.out.print("\n" + "Please enter video path directory: ");

        String modelWeights = "C:/Users/yaamin/Downloads/weights/braintiny2.weights";
        String modelConfig = "C:/Users/yaamin/Downloads/cfg/braintiny.cfg";
        String filePath = path.nextLine();
        
       
        
        VideoCapture capture = new VideoCapture(filePath);
        Mat frame = new Mat();

        JFrame jframe = new JFrame("Video");
        JLabel vidpanel = new JLabel();
        jframe.setContentPane(vidpanel);
        jframe.setSize(800, 800);
        jframe.setVisible(true);

        Net net = Dnn.readNetFromDarknet(modelConfig, modelWeights);

        
        
        Size sz = new Size(416, 416);

        List<Mat> result = new ArrayList<>();
        List<String> outBlobNames = getOutputNames(net);

        while (true) {

            if (capture.read(frame)) {

                Mat blob = Dnn.blobFromImage(frame, 1 / 255.0, sz, new Scalar(0), true, false);
                net.setInput(blob);
                net.forward(result, outBlobNames);

                float confThreshold = 0.5f;
                List<Integer> clsIds = new ArrayList<>();
                List<Float> confs = new ArrayList<>();
                List<Rect2d> rects = new ArrayList<>();
                for (int i = 0; i < result.size(); ++i) {

                    Mat level = result.get(i);
                    for (int j = 0; j < level.rows(); ++j) {
                        Mat row = level.row(j);
                        Mat scores = row.colRange(5, level.cols());
                        Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                        float confidence = (float) mm.maxVal;
                        Point classIdPoint = mm.maxLoc;

                        if (confidence > confThreshold) {
                            int centerX = (int) (row.get(0, 0)[0] * frame.cols());
                            int centerY = (int) (row.get(0, 1)[0] * frame.rows());
                            int width = (int) (row.get(0, 2)[0] * frame.cols());
                            int height = (int) (row.get(0, 3)[0] * frame.rows());
                            int left = centerX - width / 2;
                            int top = centerY - height / 2;

                            clsIds.add((int) classIdPoint.x);
                            confs.add((float) confidence);
                            rects.add(new Rect2d(left, top, width, height));
                        }

                    }

                }
                float nmsThresh = 0.4f;
                MatOfFloat confidence = new MatOfFloat(Converters.vector_float_to_Mat(confs));
                Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);
                MatOfRect2d boxes = new MatOfRect2d();
                boxes.fromList(rects);
                MatOfInt indices = new MatOfInt();
                Dnn.NMSBoxes(boxes, confidence, confThreshold, nmsThresh, indices);

                int[] ind = indices.toArray();

                for (int i = 0; i < ind.length; ++i) {
                    int idx = (int) indices.get(i, 0)[0];
                    Rect2d box = boxesArray[idx];
                    Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0, 0, 255), 4);
                    Imgproc.putText(frame, "Tumor", box.tl(), Imgproc.FONT_HERSHEY_COMPLEX, 1, new Scalar(0, 0, 255),
                            3);

                }

                ImageIcon image = new ImageIcon(Mat2bufferedImage(frame));
                vidpanel.setIcon(image);
                vidpanel.repaint();

            }
        }
    }

    private static BufferedImage Mat2bufferedImage(Mat image) {
        MatOfByte bytemat = new MatOfByte();
        Imgcodecs.imencode(".jpg", image, bytemat);
        byte[] bytes = bytemat.toArray();
        InputStream in = new ByteArrayInputStream(bytes);
        BufferedImage img = null;
        try {
            img = ImageIO.read(in);
        } catch (IOException e) {

            e.printStackTrace();
        }
        return img;
    }

    public static void image() throws IOException {
        Scanner path = new Scanner(System.in);
        System.out.print("\n" + "Please enter image path directory: ");

        Mat img = Imgcodecs.imread(path.nextLine());

        Net net = Dnn.readNetFromDarknet("C:/Users/yaamin/Downloads/cfg/braintiny.cfg",
                "C:/Users/yaamin/Downloads/weights/braintiny2.weights");
        
       
        DetectionModel model = new DetectionModel(net);
        model.setInputParams(1 / 255.0, new Size(416, 416), new Scalar(0), true);

        MatOfInt classIds = new MatOfInt();
        MatOfFloat scores = new MatOfFloat();
        MatOfRect boxes = new MatOfRect();
        
        model.detect(img, classIds, scores, boxes, 0.4f, 0.5f);

        for (int i = 0; i < classIds.rows(); i++) {
            Rect box = new Rect(boxes.get(i, 0));
            Imgproc.rectangle(img, box, new Scalar(0, 255, 0), 2);
            Imgproc.putText(img, "Tumor", new Point(box.x, box.y - 5),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);
        }
        
        HighGui.imshow("Image", img);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();
        
        System.exit(0);
    }
}
