package com.trustnet.backend.controller;

import com.trustnet.backend.entity.User;
import com.trustnet.backend.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/user")

public class UserController {
    @Autowired
    private UserService userService;

    @PostMapping("/register")
    public String register(@RequestBody User user) {
        return userService.registerUser(user);
    }

    @PostMapping("/login")
    public String login(@RequestBody User user){
     return userService.loginUser(user);  // ✅ call service method instead of using repo directly
    }

}
