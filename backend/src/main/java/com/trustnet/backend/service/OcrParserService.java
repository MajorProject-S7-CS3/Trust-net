package com.trustnet.backend.service;

import org.springframework.stereotype.Service;
import java.util.regex.*;

@Service
public class OcrParserService {

    public String extractAadhaarNumber(String text) {
        Pattern pattern = Pattern.compile("\\b\\d{4}\\s\\d{4}\\s\\d{4}\\b");
        Matcher matcher = pattern.matcher(text);
        return matcher.find() ? matcher.group() : null;
    }

    public String extractDob(String text) {
        Pattern pattern = Pattern.compile("\\b\\d{2}/\\d{2}/\\d{4}\\b");
        Matcher matcher = pattern.matcher(text);
        return matcher.find() ? matcher.group() : null;
    }

    public String extractGender(String text) {
        if (text.toLowerCase().contains("male")) return "Male";
        if (text.toLowerCase().contains("female")) return "Female";
        return null;
    }

    public String extractName(String text) {
        String[] lines = text.split("\\n");
        for (String line : lines) {
            if (line.matches("^[A-Za-z ]{3,}$") &&
                !line.toLowerCase().contains("government") &&
                !line.toLowerCase().contains("india")) {
                return line.trim();
            }
        }
        return null;
    }
}
