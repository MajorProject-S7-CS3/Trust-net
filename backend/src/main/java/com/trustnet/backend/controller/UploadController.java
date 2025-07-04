package com.trustnet.backend.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.trustnet.backend.model.UploadResponse;
import com.trustnet.backend.service.UploadService;

@RestController
@RequestMapping("/api/upload")
public class UploadController {

    @Autowired
    private UploadService uploadService;

    @PostMapping("/document")
    public ResponseEntity<UploadResponse> uploadDocument(
            @RequestParam("document") MultipartFile document,
            @RequestParam("selfie") MultipartFile selfie) {

        UploadResponse response = uploadService.storeFiles(document, selfie);
        return ResponseEntity.ok(response);
    }
}
