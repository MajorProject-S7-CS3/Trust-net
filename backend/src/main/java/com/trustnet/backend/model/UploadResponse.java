package com.trustnet.backend.model;

public class UploadResponse {
    private String status;
    private String documentName;
    private String selfieName;

    public UploadResponse(String status, String documentName, String selfieName) {
        this.status = status;
        this.documentName = documentName;
        this.selfieName = selfieName;
    }

    public String getStatus() {
        return status;
    }

    public String getDocumentName() {
        return documentName;
    }

    public String getSelfieName() {
        return selfieName;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public void setDocumentName(String documentName) {
        this.documentName = documentName;
    }

    public void setSelfieName(String selfieName) {
        this.selfieName = selfieName;
    }
}
// This class represents the response returned after a file upload operation, containing the status and names of the uploaded files.