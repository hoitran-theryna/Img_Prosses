Hướng Dẫn Cài Đặt Android Studio và Import Thư Viện OpenCV
1. Tải và Cài Đặt Android Studio

Truy cập trang chính thức của Android Studio: https://developer.android.com/studio.
Nhấn nút Download Android Studio để tải phiên bản mới nhất.
Chạy file cài đặt vừa tải và làm theo hướng dẫn trên màn hình để hoàn tất cài đặt.
Sau khi cài đặt, mở Android Studio và thiết lập môi trường phát triển (bao gồm cài đặt SDK nếu cần).

2. Tải Thư Viện OpenCV

Truy cập trang tải OpenCV cho Android: https://opencv.org/releases/.
Tải phiên bản OpenCV mới nhất (ví dụ: OpenCV 4.x.x).
Giải nén file vừa tải (thường là file .zip) vào một thư mục dễ nhớ trên máy của bạn.

3. Import Thư Viện OpenCV vào Android Studio

Mở dự án Android Studio của bạn.
Trong phần Project Structure:
Chọn File > Project Structure.
Nhấn vào Dependencies tab.
Nhấn + và chọn Module Dependency.
Chọn thư mục OpenCV đã giải nén (thường là sdk/java trong thư mục OpenCV) và thêm nó vào dự án.


Cập nhật file build.gradle (Module: app) bằng cách thêm dòng sau vào phần dependencies:implementation files('libs/opencv.jar')


Đồng bộ hóa dự án bằng cách nhấn Sync Project with Gradle Files.

4. Cấu Hình OpenCV trong Dự Án

Sao chép thư mục libs từ thư mục OpenCV đã giải nén vào thư mục app/src/main trong dự án của bạn.
Trong file AndroidManifest.xml, thêm quyền truy cập camera nếu cần:<uses-permission android:name="android.permission.CAMERA" />


Khởi tạo OpenCV trong ứng dụng (ví dụ trong MainActivity.java):import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Không thể khởi tạo OpenCV!");
        } else {
            Log.d("OpenCV", "OpenCV khởi tạo thành công!");
        }
    }
}



5. Xây Dựng và Chạy Ứng Dụng

Xây dựng dự án bằng cách nhấn Build > Make Project.
Kết nối thiết bị Android hoặc sử dụng emulator.
Chạy ứng dụng bằng cách nhấn Run 'app'.

6. Lưu Ý

Đảm bảo kích thước tệp OpenCV (như libopencv_java4.so) không vượt quá giới hạn của hệ thống quản lý phiên bản (ví dụ: 10 MB trên một số nền tảng).
Nếu gặp lỗi, kiểm tra lại đường dẫn thư viện hoặc phiên bản OpenCV tương thích với Android Studio.

Happy coding! 🚀
