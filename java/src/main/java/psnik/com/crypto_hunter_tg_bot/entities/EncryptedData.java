package psnik.com.crypto_hunter_tg_bot.entities;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
public class EncryptedData {

    @JsonProperty("iv")
    private String iv;

    @JsonProperty("ct")
    private String ct;
}
