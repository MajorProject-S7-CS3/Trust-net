package com.trustnet.backend.service;

import com.trustnet.backend.model.UploadResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.*;

@Service
public class UploadService {

    public UploadResponse storeFiles(MultipartFile document, MultipartFile selfie) {
        try {
            String docName = Path.of(document.getOriginalFilename()).getFileName().toString();
            String selfieName = Path.of(selfie.getOriginalFilename()).getFileName().toString();

            Path docPath = Paths.get("uploads/docs/" + docName);
            Path selfiePath = Paths.get("uploads/selfies/" + selfieName);

            Files.createDirectories(docPath.getParent());
            Files.createDirectories(selfiePath.getParent());

            Files.write(docPath, document.getBytes(), StandardOpenOption.CREATE);
            Files.write(selfiePath, selfie.getBytes(), StandardOpenOption.CREATE);

            return new UploadResponse("Success", docName, selfieName);
        } catch (IOException e) {
            return new UploadResponse("Failure", null, null);
        }
    }
}
// This service handles the file upload logic, storing the files in specified directories and returning an UploadResponse object.