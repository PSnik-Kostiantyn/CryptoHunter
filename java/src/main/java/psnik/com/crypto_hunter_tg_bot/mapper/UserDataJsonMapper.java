package psnik.com.crypto_hunter_tg_bot.mapper;

import com.fasterxml.jackson.databind.ObjectMapper;
import psnik.com.crypto_hunter_tg_bot.entities.EncryptedData;

public class UserDataJsonMapper {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static String toJson(EncryptedData userData) {
        try {
            return objectMapper.writeValueAsString(userData);
        } catch (Exception e) {
            throw new RuntimeException("Failed to convert UserData to JSON", e);
        }
    }
}
