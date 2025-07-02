package com.trustnet.backend.service;

import com.trustnet.backend.entity.User;
import com.trustnet.backend.model.Role;
import com.trustnet.backend.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class UserService {
    @Autowired
    private PasswordEncoder passwordEncoder;
    @Autowired
    private UserRepository userRepo;

    public String registerUser(User user) {
        if (userRepo.existsByEmail(user.getEmail())) {
            return "Email already registered";
        }
        user.setPassword(passwordEncoder.encode(user.getPassword())); // encrypt password
        if (user.getRole() == null) {
        user.setRole(Role.USER); // default role
        }
        userRepo.save(user);
        return "User registered successfully";
    }
    public String loginUser(User loginData) {
    User user = userRepo.findByEmail(loginData.getEmail());
    if (user == null) {
        return "User not found";
    }

    if (passwordEncoder.matches(loginData.getPassword(), user.getPassword())) {
        return "Login successful as " + user.getRole();
    } else {
        return "Invalid password";
    }
}

}
