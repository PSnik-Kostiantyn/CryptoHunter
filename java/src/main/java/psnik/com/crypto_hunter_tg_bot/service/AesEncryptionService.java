package psnik.com.crypto_hunter_tg_bot.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;
import org.springframework.stereotype.Component;
import psnik.com.crypto_hunter_tg_bot.entities.EncryptedData;

import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.Base64;

@Component
@PropertySource("classpath:application.properties")
public class AesEncryptionService {

    private static final String ALGORITHM = "AES/CBC/PKCS5Padding";

    @Value("${secret.key}")
    String secretKey;

    public EncryptedData encryptData(String data) throws Exception {
        SecretKey key = new SecretKeySpec(secretKey.getBytes(StandardCharsets.UTF_8), "AES");
        Cipher cipher = Cipher.getInstance(ALGORITHM);

        byte[] iv = new byte[16];
        new SecureRandom().nextBytes(iv);
        IvParameterSpec ivSpec = new IvParameterSpec(iv);

        cipher.init(Cipher.ENCRYPT_MODE, key, ivSpec);

        byte[] encrypted = cipher.doFinal(data.getBytes(StandardCharsets.UTF_8));

        return EncryptedData.builder()
                .iv(Base64.getEncoder().encodeToString(iv))
                .ct(Base64.getEncoder().encodeToString(encrypted))
                .build();
    }

    public String decryptData(String ivBase64, String ctBase64) throws Exception {
        SecretKey key = new SecretKeySpec(secretKey.getBytes(StandardCharsets.UTF_8), "AES");
        Cipher cipher = Cipher.getInstance(ALGORITHM);

        IvParameterSpec ivSpec = new IvParameterSpec(Base64.getDecoder().decode(ivBase64));
        cipher.init(Cipher.DECRYPT_MODE, key, ivSpec);

        byte[] decryptedBytes = cipher.doFinal(Base64.getDecoder().decode(ctBase64));
        return new String(decryptedBytes, StandardCharsets.UTF_8);
    }
}
