package psnik.com.crypto_hunter_tg_bot.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import psnik.com.crypto_hunter_tg_bot.entities.EncryptedData;
import psnik.com.crypto_hunter_tg_bot.mapper.UserDataJsonMapper;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

@Service
@PropertySource("classpath:application.properties")
public class UserService {

    private final RestTemplate restTemplate;
    private final AesEncryptionService aesEncryptionService;

    @Value("${python.server}")
    private String serverUrl;

    @Autowired
    public UserService(RestTemplate restTemplate, AesEncryptionService aesEncryptionService) {
        this.restTemplate = restTemplate;
        this.aesEncryptionService = aesEncryptionService;
    }

    public ResponseEntity<String> postNumberByUserID(String telegramUserID, int ifBelow) {
        try {
            Map<String, Object> userObject = new HashMap<>();
            userObject.put("telegram_id", telegramUserID);
            userObject.put("if_below", ifBelow);

            ObjectMapper objectMapper = new ObjectMapper();
            String jsonArray = objectMapper.writeValueAsString(Collections.singletonList(userObject));

            EncryptedData userData = aesEncryptionService.encryptData(jsonArray);
            HttpHeaders headers = new HttpHeaders();
            headers.set("Content-Type", "application/json");

            String jsonBody = UserDataJsonMapper.toJson(userData);
            HttpEntity<String> request = new HttpEntity<>(jsonBody, headers);

            return restTemplate.postForEntity(serverUrl, request, String.class);

        } catch (Exception e) {
            throw new RuntimeException("Failed to encrypt or send data", e);
        }
    }
}
