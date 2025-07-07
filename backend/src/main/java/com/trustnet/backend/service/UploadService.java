package com.trustnet.backend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.trustnet.backend.model.UploadResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
//import org.springframework.http.client.MultipartBodyBuilder;
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
    @Autowired
    private OcrParserService ocrParser;

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

            // Send saved document to OCR microservice
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

            // Return response with OCR data
            return new UploadResponse("Success", docName, selfieName, ocrText);

        } catch (IOException e) {
            e.printStackTrace();
            return new UploadResponse("Failure", null, null, null);
        }
    }

    // Send the saved file to OCR microservice via HTTP POST
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
}
