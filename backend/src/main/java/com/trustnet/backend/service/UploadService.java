package com.trustnet.backend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.trustnet.backend.model.UploadResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.nio.file.*;

@Service
public class UploadService {

    @Autowired
    private WebClient.Builder webClientBuilder;

    public UploadResponse storeFiles(MultipartFile document, MultipartFile selfie) {
        try {
            // Extract filenames
            String docName = Path.of(document.getOriginalFilename()).getFileName().toString();
            String selfieName = Path.of(selfie.getOriginalFilename()).getFileName().toString();

            // Define paths
            Path docPath = Paths.get("uploads/docs/" + docName);
            Path selfiePath = Paths.get("uploads/selfies/" + selfieName);

            // Create directories if not present
            Files.createDirectories(docPath.getParent());
            Files.createDirectories(selfiePath.getParent());

            // Save files locally
            Files.write(docPath, document.getBytes(), StandardOpenOption.CREATE);
            Files.write(selfiePath, selfie.getBytes(), StandardOpenOption.CREATE);

            // ✅ Perform face match
            boolean faceMatch = verifyFaceMatch(docPath, selfiePath);
            if (!faceMatch) {
                return new UploadResponse("Face mismatch", docName, selfieName, null);
            }

            // ✅ Perform OCR
            String ocrText = extractTextFromDocument(docPath);

            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(ocrText);

            String name = root.path("name").asText(null);
            String aadhaar = root.path("aadhaar_number").asText(null);
            String gender = root.path("gender").asText(null);
            String dob = root.path("dob").asText(null);

            System.out.println("Extracted Fields:");
            System.out.println("Name: " + name);
            System.out.println("DOB: " + dob);
            System.out.println("Gender: " + gender);
            System.out.println("Aadhaar: " + aadhaar);

            return new UploadResponse("Success", docName, selfieName, ocrText);

        } catch (IOException e) {
            e.printStackTrace();
            return new UploadResponse("Failure", null, null, null);
        }
    }

    // 🔍 OCR Microservice Call
    public String extractTextFromDocument(Path docPath) throws IOException {
        byte[] fileBytes = Files.readAllBytes(docPath);

        return webClientBuilder.build()
                .post()
                .uri("http://localhost:5000/ocr")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData("file", new ByteArrayResource(fileBytes) {
                    @Override
                    public String getFilename() {
                        return docPath.getFileName().toString();
                    }
                }))
                .retrieve()
                .bodyToMono(String.class)
                .block(); // waits for response
    }

    // 🧠 Face Match Microservice Call
    public boolean verifyFaceMatch(Path docPath, Path selfiePath) {
        try {
            MultipartBodyBuilder builder = new MultipartBodyBuilder();

            builder.part("document", new ByteArrayResource(Files.readAllBytes(docPath)) {
                @Override
                public String getFilename() {
                    return docPath.getFileName().toString();
                }
            }).contentType(MediaType.IMAGE_JPEG);

            builder.part("selfie", new ByteArrayResource(Files.readAllBytes(selfiePath)) {
                @Override
                public String getFilename() {
                    return selfiePath.getFileName().toString();
                }
            }).contentType(MediaType.IMAGE_JPEG);

            String response = webClientBuilder.build()
                    .post()
                    .uri("http://localhost:5000/face-match")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(builder.build()))
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            ObjectMapper mapper = new ObjectMapper();
            JsonNode json = mapper.readTree(response);
            boolean isMatch = json.has("match") && json.get("match").asBoolean(false);
            double confidence = json.has("confidence") ? json.get("confidence").asDouble() : 0.0;
            System.out.println("DeepFace match: " + isMatch + ", confidence: " + confidence);

            return isMatch;

        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
}
