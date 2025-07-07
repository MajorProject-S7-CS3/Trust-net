package com.trustnet.backend.model;

public class UploadResponse {

    private String status;
    private String documentName;
    private String selfieName;
    private String ocrText;

    public UploadResponse() {
    }

    public UploadResponse(String status, String documentName, String selfieName, String ocrText) {
        this.status = status;
        this.documentName = documentName;
        this.selfieName = selfieName;
        this.ocrText = ocrText;
    }

    // Getters and Setters

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getDocumentName() {
        return documentName;
    }

    public void setDocumentName(String documentName) {
        this.documentName = documentName;
    }

    public String getSelfieName() {
        return selfieName;
    }

    public void setSelfieName(String selfieName) {
        this.selfieName = selfieName;
    }

    public String getOcrText() {
        return ocrText;
    }

    public void setOcrText(String ocrText) {
        this.ocrText = ocrText;
    }
}