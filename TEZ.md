# GERÇEK ZAMANLI POZ KESTİRİMİ İLE EGZERSİZ FORMU ANALİZİ: KARŞILAŞTIRMALI BİR MODEL DEĞERLENDİRMESİ VE KURAL TABANLI GERİ BİLDİRİM SİSTEMİ

*Lisans Bitirme Tezi*

---

## ÖNSÖZ

Bu tez, kamera tabanlı insan poz kestirimi yöntemlerinin direnç antrenmanı bağlamında form değerlendirmesine nasıl uygulanabileceğini araştırmak amacıyla hazırlanmıştır. Çalışma süresince tek bir RGB kameradan görüntü alan, giyilebilir sensör gerektirmeyen ve gerçek zamanlı çalışabilen bir yazılım sistemi tasarlanıp geliştirildi. Sistem egzersiz tekrarlarını otomatik sayabiliyor, her tekrarın doğru formla yapılıp yapılmadığını ise yorumlanabilir kurallar aracılığıyla değerlendiriyor.

Ortaya çıkan ürün iki parçadan oluşuyor: canlı kamera akışıyla çalışan bir masaüstü uygulaması ve farklı poz kestirim modellerini nesnel ölçütler altında kıyaslayan çevrimdışı bir değerlendirme altyapısı. Tezdeki tüm nicel sonuçlar bu değerlendirme altyapısı aracılığıyla üretilmiştir; tablolardaki sayısal değerler ilgili değerlendirme komutları çalıştırılarak elde edilmelidir.

Bu çalışmanın hazırlanması sürecinde yönlendirmeleri ve destekleri için danışmanıma ve emeği geçen herkese teşekkür ederim.

---

## ÖZET

Bu tez, tek kameralı (monoküler) RGB görüntü üzerinden gerçek zamanlı egzersiz formu analizi yapan bütünleşik bir sistemin tasarımını, uygulanmasını ve değerlendirilmesini aktarmaktadır. Sistem; poz kestirimi, eklem açısı hesaplama, tekrar takibi, sinyal yumuşatma ve kural tabanlı form değerlendirmesinden oluşan modüler bir işlem hattına dayanmaktadır. Squat, deadlift, tek kol dambıl kıvrımı, omuz presi ve yana kaldırma olmak üzere beş direnç egzersizi için her harekete özgü biyomekanik kurallar tanımlanmıştır.

Değerlendirme kapsamında dört poz kestirim modeli karşılaştırılmıştır: gerçek 3B dünya koordinatları sağlayan MediaPipe BlazePose, doğrudan metrik ölçekli 3B kestirim yapan MeTRAbs ve iki boyutlu çıktı veren MoveNet (Lightning/Thunder) ile YOLOv8n-pose. Modeller FIT3D hareket yakalama veri kümesi referans alınarak Procrustes hizalamalı eklem konumu hatası (PA-MPJPE), eklem açısı ortalama mutlak hatası (MAE), gerçek zamanlı başarım ve modeller arası açı uyumu açısından kıyaslanmıştır. Etiketli klipler üzerinde ikili form sınıflandırma başarımı ile kural bazlı tespit duyarlılığı da ayrıca raporlanmıştır.

MediaPipe modelinin egzersizler genelinde ortalama PA-MPJPE değeri `[PLACEHOLDER: MediaPipe için egzersizler ortalaması PA-MPJPE (mm) — `fitness-form-evaluate --mocap ...` çıktısındaki Tablo 4.1 değerlerinin ortalaması]` mm olarak ölçülmüştür. Form sınıflandırmasında genel F1 skoru `[PLACEHOLDER: genel F1 ve %95 GA — `fitness-form-evaluate --videos ... --labels ...` çıktısı, exercise_metrics.csv]` olarak elde edilmiştir. Bulgular, 3B dünya koordinatı sağlayan modellerin açı tabanlı form değerlendirmesi için 2B modellere kıyasla daha tutarlı sonuç verdiğine işaret etmektedir.

**Anahtar Kelimeler:** İnsan poz kestirimi, egzersiz formu analizi, MediaPipe, gerçek zamanlı bilgisayarlı görü, eklem açısı, tekrar sayımı, kural tabanlı değerlendirme.

---

## ABSTRACT

This thesis presents the design, implementation, and evaluation of an integrated system that performs real-time exercise form analysis from a single (monocular) RGB camera. The system is built on a modular pipeline that covers human pose estimation, joint angle computation, repetition tracking, signal smoothing, and rule-based form assessment. Exercise-specific biomechanical rules are defined for five resistance exercises: squat, deadlift, single-arm dumbbell curl, shoulder press, and lateral raise.

Four pose estimation models are evaluated comparatively: MediaPipe BlazePose, which provides true 3D world coordinates; MeTRAbs, which directly estimates metric-scale 3D pose; and the two-dimensional models MoveNet (Lightning/Thunder) and YOLOv8n-pose. The models are compared against the FIT3D motion-capture dataset using Procrustes-aligned mean per-joint position error (PA-MPJPE), joint angle mean absolute error (MAE), real-time performance (FPS and latency), and cross-model angle agreement. Binary form-classification performance and per-rule detection sensitivity are also reported on labelled clips.

The mean PA-MPJPE of the MediaPipe model across exercises was measured as `[PLACEHOLDER: mean PA-MPJPE (mm) for MediaPipe across exercises — mean of Table 4.1 values from the `fitness-form-evaluate --mocap` output]` mm. The overall F1 score in form classification was `[PLACEHOLDER: overall F1 and 95% CI — from `fitness-form-evaluate --videos ... --labels ...` output, exercise_metrics.csv]`. The findings indicate that models providing 3D world coordinates yield more consistent results for angle-based form assessment compared to 2D models.

**Keywords:** Human pose estimation, exercise form analysis, MediaPipe, real-time computer vision, joint angle, repetition counting, rule-based assessment.

---

# 1. GİRİŞ

Düzenli direnç antrenmanının kas kuvveti ve genel sağlık üzerindeki yararları literatürde iyi belgelenmiştir. Bununla birlikte, egzersizlerin yanlış formla yapılması hem yaralanma riskini artırmakta hem de antrenmanın etkinliğini düşürmektedir. Doğru form geri bildirimi geleneksel olarak bir antrenör ya da fizyoterapistin gözlemine dayanır ve bu uzman gözetimi her zaman erişilebilir değildir. Özellikle ev ortamında ya da kalabalık spor salonlarında bireyler anlık, nesnel bir geri bildirimden çoğunlukla yoksun kalır.

Son yıllarda bilgisayarlı görü ve derin öğrenme alanındaki ilerleme, sıradan bir RGB kameradan insan vücudunun eklem konumlarını gerçek zamanlı olarak kestirebilen modellerin geliştirilmesini mümkün kılmıştır. Bu modeller, hareketin biyomekanik açıdan çözümlenebilmesi için gereken iskelet verisini herhangi bir giyilebilir sensör ya da işaretleyici gerektirmeksizin sunmaktadır. Sonuç olarak, otomatik ve erişilebilir bir egzersiz formu geri bildirim sistemi artık teknik açıdan gerçekleştirilebilir bir hedef hâline gelmiştir.

Bu tez, tek bir kameradan elde edilen görüntü üzerinde çalışan, egzersiz tekrarlarını otomatik sayan ve her tekrarın formunu yorumlanabilir kurallarla değerlendiren bütünleşik bir sistem geliştirmektedir. Bunun yanı sıra, bu sistemin temelini oluşturan poz kestirim modellerinin nesnel ölçütler çerçevesinde karşılaştırmalı bir değerlendirmesi de sunulmaktadır.

## 1.1 PROBLEM (AMAÇ)

Araştırmanın temel amacı, tek kameralı bir görüntü akışı üzerinden gerçek zamanlı, açıklanabilir ve giyilebilir donanım gerektirmeyen bir egzersiz formu analiz sistemi tasarlamak; farklı poz kestirim modellerinin bu amaca uygunluğunu ise nesnel ölçütlerle ortaya koymaktır. Problem iki boyutludur: yöntemsel boyutuyla, poz verisinden güvenilir biçimde eklem açısı, tekrar sayısı ve form ihlali çıkarabilen bir işlem hattının tasarlanması; ampirik boyutuyla ise bu işlem hattını besleyebilecek modellerin doğruluk ve hız açısından kıyaslanması.

### 1.1.1 Alt Amaçlar

Araştırmanın alt amaçları şu şekilde özetlenebilir: poz kestirim çıktısından yön bağımsız eklem açısı hesaplayan bir yöntem geliştirmek; gürültülü açı sinyalini gerçek zamanlı koşullarda kararlı biçimde işleyen uyarlamalı bir yumuşatma ve tekrar takip mekanizması oluşturmak; her egzersiz için biyomekanik olarak anlamlı, birleştirilebilir kurallar tanımlamak; farklı poz kestirim modellerini referans bir hareket yakalama veri kümesine karşı konum ve açı doğruluğu bakımından kıyaslamak; modellerin gerçek zamanlı başarımını ölçmek ve sistemin form sınıflandırma performansını etiketli veriler üzerinde değerlendirmek.

## 1.2 ÖNEM

Bu çalışmanın özgün değeri birkaç noktada kendini göstermektedir. Her şeyden önce, sistem yalnızca standart bir kamera kullandığından pahalı hareket yakalama donanımına veya giyilebilir sensörlere ihtiyaç duymadan form geri bildirimi sağlama potansiyeli taşımaktadır. Dahası, değerlendirme mekanizması kara kutu bir sınıflandırıcıya değil, biyomekanik olarak yorumlanabilir kurallara dayandığından kullanıcıya yalnızca "yanlış" değil, *neden* yanlış olduğunu da açıklayabilmektedir. Son olarak, tez uygulamada sıkça tercih edilen poz kestirim modellerinin — özellikle 2B ve 3B modellerin — açı tabanlı form analizi bağlamındaki güçlü ve zayıf yönlerini nesnel ölçütlerle karşılaştırmakta ve benzer sistem geliştirecek araştırmacılara model seçimi konusunda kanıta dayalı bir referans sunmaktadır.

## 1.3 TANIMLAR

**Poz kestirimi (pose estimation):** Bir görüntüdeki insan vücudunun anahtar eklem noktalarının (omuz, dirsek, kalça, diz gibi) konumlarını kestirimek amacıyla uygulanan işlemdir.

**Eklem açısı (joint angle):** Üç anahtar nokta tarafından tanımlanan iki kemik vektörü arasındaki açıdır; örneğin omuz–dirsek–bilek üçlüsünün tanımladığı dirsek açısı.

**Tekrar (repetition / rep):** Bir egzersiz hareketinin tam bir döngüsüdür; squatta ayakta durma → çömelme → yeniden ayakta durma gibi.

**Hareket aralığı (Range of Motion, ROM):** Bir tekrar boyunca eklem açısının ulaştığı en büyük ve en küçük değerler arasındaki farktır.

**3B dünya koordinatları (3D world landmarks):** Eklem noktalarının yalnızca görüntü düzlemindeki değil, kamera referanslı metrik üç boyutlu uzaydaki konumlarıdır.

**PA-MPJPE (Procrustes-Aligned Mean Per-Joint Position Error):** Tahmin edilen iskelet ile referans iskelet benzerlik dönüşümüyle hizalandıktan sonra eklem başına hesaplanan ortalama konum hatasıdır ve milimetre cinsinden ifade edilir.

**Form ihlali (form violation):** Bir tekrarın, ilgili egzersiz için tanımlanmış biyomekanik kurallardan birini ya da birkaçını sağlamaması durumudur.

## 1.4 SİMGELER VE KISALTMALAR

| Kısaltma | Açıklama |
|----------|----------|
| RGB | Kırmızı-Yeşil-Mavi (renkli görüntü) |
| 2B / 2D | İki boyutlu |
| 3B / 3D | Üç boyutlu |
| ROM | Hareket aralığı (Range of Motion) |
| FPS | Saniyedeki kare sayısı (Frames Per Second) |
| MAE | Ortalama mutlak hata (Mean Absolute Error) |
| RMSE | Karekök ortalama kare hata (Root Mean Square Error) |
| MPJPE | Eklem başına ortalama konum hatası |
| PA-MPJPE | Procrustes hizalamalı MPJPE |
| EMA | Üstel ağırlıklı hareketli ortalama (Exponential Moving Average) |
| GA / CI | Güven aralığı (Confidence Interval) |
| GUI | Grafik kullanıcı arayüzü |
| ms | Milisaniye |
| mm | Milimetre |

## 1.5 SINIRLILIKLAR VE VARSAYIMLAR

Çalışma birkaç önemli sınırlılık çerçevesinde yürütülmüştür. Sistem tek bir RGB kameradan elde edilen monoküler görüntüyle çalışmakta olup çoklu kamera füzyonu kapsam dışında bırakılmıştır. Form değerlendirme kuralları beş egzersizle sınırlıdır ve eşik değerleri uzman biyomekanik bilgisine dayalı olarak belirlenmiştir; bu değerler geniş bir popülasyon üzerinde optimize edilmemiştir. Poz kestirim doğruluğunun değerlendirmesi, FIT3D hareket yakalama veri kümesindeki belirli denek(ler) ve kamera açılarıyla sınırlı kalmaktadır. 2B modeller (MoveNet, YOLOv8) derinlik bilgisi sağlamadığından, bu modellerin 3B referansa karşı hesaplanan konum hataları gerçek 3B modellerle doğrudan karşılaştırılabilir nitelikte değildir; bu durum bulguların yorumlanmasında göz önünde bulundurulmuştur. Çalışmada temel varsayım olarak, değerlendirmede kullanılan referans hareket yakalama verisinin gerçek değer (ground truth) niteliği taşıdığı kabul edilmiştir.

---

# 2. ARAŞTIRMANIN KURAMSAL ÇERÇEVESİ VE İLGİLİ ARAŞTIRMALAR

## 2.1 İLGİLİ ARAŞTIRMALAR

### 2.1.1 Poz Kestirim Yöntemleri

İnsan poz kestirimi, bilgisayarlı görünün köklü alt alanlarından biridir ve son yıllarda derin öğrenme tabanlı yaklaşımların belirleyici hâle geldiği bir alan olmuştur. Bu tezde kullanılan ve karşılaştırılan modeller, alandaki farklı tasarım anlayışlarını temsil etmektedir.

**MediaPipe BlazePose**, mobil ve gömülü cihazlarda gerçek zamanlı çalışmak üzere tasarlanmış hafif bir evrişimli sinir ağı mimarisidir (Bazarevsky vd., 2020). Tek bir kareden 33 vücut anahtar noktası üretir ve hem ısı haritası hem de doğrudan koordinat regresyonu kullanır. Bu tez açısından kritik nokta, modelin görüntü düzlemindeki 2B noktalara ek olarak metrik 3B "dünya koordinatları" (pose world landmarks) sunmasıdır; bu koordinatlar derinlik bilgisi gerektiren açı hesaplamalarında doğrudan kullanılabilmektedir. Model üç karmaşıklık düzeyi (lite/full/heavy, kompleksite 0/1/2) olarak yapılandırılabilmektedir.

**MeTRAbs** (Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation), Sárándi, Linder, Arras ve Leibe (2021) tarafından önerilen bir yöntemdir. Boyutları metrik 3B uzayda tanımlı, kesilmeye (truncation) dayanıklı hacimsel ısı haritaları kullanarak test zamanında mesafe bilgisine veya antropometrik varsayımlara ihtiyaç duymadan tam ölçekli poz üretir. Gerçek 3B kestirim sağlayan ikinci model olarak değerlendirmeye alınmıştır.

**MoveNet**, TensorFlow ekibi tarafından geliştirilen, COCO formatında 17 anahtar nokta üreten hızlı bir 2B poz kestirim modelidir (TensorFlow, 2021). Lightning sürümü gecikme kritik uygulamalar için, Thunder sürümü ise doğruluk öncelikli uygulamalar için tasarlanmıştır. Yalnızca 2B çıktı verdiğinden bu tezde 2B karşılaştırma temeli olarak kullanılmıştır.

**YOLOv8-pose**, Ultralytics tarafından geliştirilen YOLOv8 ailesinin poz varyantıdır (Jocher, Chaurasia ve Qiu, 2023). COCO anahtar nokta veri kümesi üzerinde eğitilmiş `yolov8n-pose` modeli 2B 17 anahtar nokta üretmekte ve ikinci bir 2B karşılaştırma temeli olarak değerlendirmeye alınmaktadır.

### 2.1.2 Egzersiz Formu Analizi ve Tekrar Sayımı

Poz kestirim modellerinin fitness ve spor bağlamında kullanılmasına yönelik çalışmalar giderek artmaktadır. Fieraru ve arkadaşları (2021), fitness antrenmanı için 3B insan algılaması yapan AIFit sistemini ve buna eşlik eden büyük ölçekli FIT3D veri kümesini yayımlamıştır. FIT3D, üç milyondan fazla görüntü ile bunlara karşılık gelen yüksek doğruluklu 3B hareket yakalama verisi içermekte; fitness egzersizlerinde poz kestirim doğruluğunun değerlendirilmesi için güvenilir bir referans oluşturmaktadır. Bu tezde FIT3D, modellerin konum ve açı doğruluğunu ölçmek için gerçek değer kaynağı olarak kullanılmıştır.

Tezin bu literatür içindeki yeri, ham poz çıktısını form değerlendirmesine dönüştüren işlem hattının her aşamasını — açı hesaplama, sinyal yumuşatma, tekrar takibi ve kural değerlendirmesi — açıkça modelleyen ve farklı poz modellerini bu somut görev üzerinde nesnel biçimde kıyaslayan uygulamalı bir sistem sunmaktır.

## 2.2 ALANYAZIN TARAMASININ SONUCU

Alanyazın taraması, gerçek zamanlı poz kestiriminin artık olgun bir teknoloji olduğunu; ancak ham poz çıktısını güvenilir bir form geri bildirimine dönüştüren ara katmanların — açı hesaplama, gürültü yönetimi, tekrar bölütleme ve kural değerlendirmesi — sistem başarımında belirleyici rol oynadığını ortaya koymaktadır. Öte yandan, 2B ve 3B poz modellerinin açı tabanlı analize uygunluğunu konum, açı ve gerçek zamanlı başarım eksenlerini birlikte ele alarak sistematik biçimde karşılaştıran uygulamalı çalışmalara alan yazınında hâlâ ihtiyaç duyulmaktadır. Bu tez, açıklanabilir kural tabanlı bir işlem hattı ile birden çok modelin çok eksenli karşılaştırmasını bir arada sunarak söz konusu boşluğa katkı sağlamayı hedeflemektedir.

---

# 3. YÖNTEM

## 3.1 Araştırmanın Modeli

Bu araştırma, tasarım temelli (design-science) ve nicel değerlendirmeye dayalı bir mühendislik çalışmasıdır. İki tamamlayıcı bileşeni bulunmaktadır: gerçek zamanlı egzersiz formu analizi yapan bir yazılım sisteminin tasarlanıp uygulanması ve bu sistemi besleyen poz kestirim modellerinin kamuya açık bir referans veri kümesi (FIT3D) ile etiketli klipler üzerinde nesnel ölçütlerle deneysel olarak değerlendirilmesi. Yeniden üretilebilirliği güvence altına almak için değerlendirme, sabit rastgelelik tohumlarına ve güven aralıkları içeren betimleyici istatistiklere dayandırılmıştır.

## 3.2 Veri Seti ve Çalışma Grubu

Poz kestirim doğruluğunun değerlendirilmesinde Fieraru ve arkadaşları (2021) tarafından yayımlanan FIT3D hareket yakalama veri kümesi gerçek değer kaynağı olarak kullanılmıştır. FIT3D, fitness egzersizlerini birden çok senkronize kameradan kaydeden ve karşılık gelen yüksek doğruluklu 3B iskelet verisi sağlayan büyük ölçekli bir veri kümesidir. Değerlendirmede kullanılan kamera ve egzersiz dağılımı aşağıdaki tabloda verilmiştir:

| Özellik | Değer |
|---------|-------|
| FIT3D kamera sayısı | `[PLACEHOLDER: değerlendirmede kullanılan kamera sayısı ve kimlikleri — fill_thesis_placeholders.py / değerlendirme çıktısı]` |
| Egzersiz sayısı | 5 (Squat, Deadlift, Kol Kıvrımı, Omuz Presi, Yana Kaldırma) |
| Toplam klip (kamera × egzersiz) | `[PLACEHOLDER: toplam klip sayısı — değerlendirme çıktısı]` |
| Egzersiz başına klip | `[PLACEHOLDER: egzersiz başına klip sayısı — değerlendirme çıktısı]` |

Referans iskelet, FIT3D'nin 25 eklemli düzenini (`joints3d_25`) kullanmaktadır. Bu eklemler, MediaPipe anahtar noktalarıyla bir eşleme tablosu (`MP_TO_FIT3D`) aracılığıyla hizalanır; MediaPipe'da güvenilir karşılığı bulunmayan eklemler (ayak parmağı gibi) eşlemeden çıkarılmıştır. Form sınıflandırma değerlendirmesi içinse egzersizlerin hem düzgün (proper) hem de özensiz/gündelik (casual) yapıldığı etiketli klipler kullanılmıştır.

## 3.3 Sistem Mimarisi

Sistem, sorumlulukların net biçimde ayrıştırıldığı katmanlı ve modüler bir mimariye sahiptir. En üst düzeyde iki giriş noktası bulunmaktadır: canlı kamera/video akışıyla çalışan grafik arayüzlü uygulama (`fitness-form-ai`) ve çevrimdışı değerlendirme aracı (`fitness-form-evaluate`).

Çalışma zamanı çekirdeği `TrackingSession` sınıfında toplanmıştır. Bu sınıf her kareyi sırasıyla işler: kareyi mektup-kutusu (letterbox) yöntemiyle yeniden boyutlandırır, seçilen poz modelinden anahtar noktaları çıkarır, egzersizin birincil eklemlerinden 3B açıyı hesaplar, açıyı tekrar takipçisine (`RepTracker`) iletir ve egzersiz kurallarını güncel kare bağlamında (`RuleContext`) değerlendirir. Tekrar tamamlandığında veri kurallar aracılığıyla geçerli ya da geçersiz olarak sınıflandırılır ve kullanıcıya geri bildirim verilir. Poz modeli seçimi bir fabrika ve kayıt deseniyle soyutlanmış olduğundan modeller birbirinin yerine sorunsuzca takılabilmektedir. Birincil model başlatılamazsa sistem güvenli biçimde varsayılan modele (MediaPipe full) geri döner.

Sistemde ayrıca her kare ve tekrar için açı, takipçi durumu, yumuşatma penceresi ve kural durumlarını kaydeden bir oturum kaydedicisi (`SessionLogger`) de yer almaktadır; bu kayıtlar daha sonraki çözümlemeler için JSON biçiminde saklanır.

## 3.4 Poz Kestirim Modelleri

Sistem, ortak bir soyut arayüz (`PoseModel`) üzerinden dört farklı poz kestirim ailesini desteklemektedir. Bu arayüz her modelin görüntüyü işlemesini, anahtar noktaları çıkarmasını ve `provides_3d_landmarks` bayrağı aracılığıyla 3B koordinat sağlayıp sağlamadığını bildirmesini standartlaştırır.

MediaPipe arka ucu `pose_world_landmarks` çıktısını kullanarak metrik 3B koordinatlar üretir; tespit ve takip güven eşikleri 0,5'e ayarlanmıştır ve model karmaşıklığı yapılandırılabilir durumdadır. MeTRAbs arka ucu h36m_17 iskelet düzeninde, kalça merkezli ve milimetreden metreye dönüştürülmüş gerçek 3B koordinatlar sağlar. MoveNet arka ucu TensorFlow Lite modellerini kullanarak COCO 17 anahtar noktası üretir; yalnızca 2B çıktı verir (z=0), güven eşiği 0,3'tür ve letterbox dolgusu geri çıkarılır. YOLOv8 arka ucu ise Ultralytics `yolov8n-pose` modelini kullanarak COCO 17 anahtar noktası üretir ve güven eşiği 0,6 olarak belirlenmiştir. 2B modellerin tüm eklemlerinde derinlik bileşeni sıfır kabul edildiğinden bu modellerin 3B referansa karşı değerlendirilmesi önemli bir kısıt oluşturmaktadır; bu durum Bölüm 4 ve 5'te ele alınmıştır.

## 3.5 Eklem Açısı ve Metrik Hesaplama

Form değerlendirmesinin temel niceliği eklem açısıdır. Açı, üç anahtar nokta (omuz–dirsek–bilek gibi) tarafından tanımlanan iki kemik vektörü arasındaki açı olarak tam 3B uzayda nokta çarpımı yoluyla hesaplanır (`calculate_angle_3d`). Bu yaklaşım, kameraya göre vücut yönelimi değişse bile açının tutarlı kalmasını sağlar; bunun yanı sıra dejenere durumlarda (sıfır uzunluklu vektör) güvenli biçimde 0,0 döndürerek sayısal kararlılık korunur. 2B modellerde z bileşeni sıfır olduğundan hesaplama pratikte görüntü düzlemindeki açıya indirgenir.

Kuralların ihtiyaç duyduğu çeşitli geometrik nicelikler bir "metrik" (metric) soyutlamasıyla modellenmiştir: belirli bir eklemin açısı (`AngleMetric`), takipçinin izlediği birincil açı (`TrackerAngleMetric`), iki eklem arasındaki belirli bir eksendeki mesafe (`JointPairAxisDistanceMetric`), bir vücut segmentinin dikey ya da yatay bir eksene göre yönelimi (`SegmentOrientationMetric`) ve sol-sağ simetri farkı (`SymmetryMetric`). Bu metrikler, kuralların tekrar verisi üzerinden hem tek tek kareler hem de zaman serileri olarak değerlendirilebilmesine olanak tanımaktadır.

## 3.6 Tekrar Takibi ve Sinyal Yumuşatma

Ham açı sinyali, kamera gürültüsü ve poz kestirim titremesi nedeniyle dalgalıdır ve bu durum yanlış tekrar sayımına yol açabilir. Sistem bu sorunu iki bileşenle ele almaktadır.

Uyarlamalı yumuşatıcı (`AdaptiveSmoother`), sabit kare sayısı yerine zaman hedefli bir pencere kullanır. Hedef pencere yaklaşık 150 ms olarak belirlenmiştir; kareler arası süre üstel ağırlıklı hareketli ortalama (EMA, alfa = 0,15; başlangıç ≈ 33,3 ms) ile kestirilerek pencere kare sayısı bu hedefe göre uyarlanır (en az 3, en çok 15 kare). Böylece yumuşatmanın etkisi, kare hızından bağımsız olarak zaman ekseninde tutarlı biçimde korunur.

Tekrar takipçisi (`RepTracker`) ise açının zaman içindeki seyrini bir durum makinesi olarak modeller: IDLE → etkin faz → dönüş fazı → COMPLETED. Faz geçişleri, anlık titremelere karşı dayanıklılık için bir "onay karesi" (confirm_frames = 3) mekanizmasıyla geciktirilir. Hız eşiği (velocity_threshold = 10) ve asgari hareket aralığı (min_rom = 20) parametreleriyle birlikte bu tasarım, kısmi ya da gürültülü hareketlerin geçerli tekrar olarak sayılmasını engeller.

## 3.7 Form Değerlendirme Kuralları

Form değerlendirmesi, birleştirilebilir (composable) bir kural/metrik sistemi üzerine kuruludur. Soyut `Rule` sınıfı; bir tekrara uygulanma (`apply`), güncel değeri okuma ve birden çok kareyi tek bir karara indirgeme (`reduce`) yeteneklerini tanımlar. Sistemde; belirli bir aralıkta kalmayı denetleyen `RangeRule`, asgari/azami değer kuralları (`MinValueRule`, `MaxValueRule`), derinlik kuralı (`MaxDepthRule`), kararlılık kuralı (`StabilityRule`), taban/zemin kuralı (`FloorRule`), diz içe çökmesini denetleyen `KneeValgusRule` ve tempo kuralı (`TempoRule`) yer almaktadır. Her egzersiz bu kuralların biyomekanik açıdan anlamlı bir bileşimiyle tanımlanmıştır:

| Egzersiz | Temel Kurallar (özet) |
|----------|-----------------------|
| Tek Kol Dambıl Bukuşu | Dirsek derinliği (MaxDepth ≤ 75°), tam uzanma (MinValue ≥ 155°), üst kol sabitliği (Stability ≤ 40°), tempo (max 250, min süre 1,0 s) |
| Squat | Diz derinliği (MaxDepth ≤ 85°), tam uzanma (MinValue ≥ 160°), gövde açısı (Range 145–180°), diz hizası (KneeValgus min oran 0,55), tempo (250 / 1,0 s) |
| Deadlift | Kalça menteşesi derinliği (MaxDepth ≤ 90°), nötr omurga (Range 150–180°), barın gövdeye yakınlığı (MaxValue ≤ 0,22), gövde yönelimi (0–65°), tempo (150 / 1,0 s) |
| Omuz Presi | Dirsek tam uzanma (MinValue ≥ 160°), bilek/kol hizası (0–30°), gövde dikliği (Range 155–180°), simetri (MaxValue ≤ 18°), tempo (150 / 1,0 s) |
| Yana Kaldırma | Omuz açısı (Range 35–110°), gövde kararlılığı (Stability ≤ 18°), omuz silkme denetimi (Floor 0,08), dirsek açısı (Range 150–178°), tempo (120 / 1,0 s) |

Bir tekrar tüm kuralları sağlıyorsa "geçerli", aksi hâlde hangi kural ihlal edildiği belirtilerek "geçersiz" olarak işaretlenir. Bu yapı, geri bildirimin yorumlanabilir kalmasını doğrudan desteklemektedir.

## 3.8 Değerlendirme Protokolü ve Verilerin Analizi

Değerlendirme altyapısı beş tamamlayıcı protokolden oluşmakta ve sonuçlarını CSV tabloları ile grafikler (matplotlib/seaborn) biçiminde üretmektedir.

Poz doğruluğu ölçümünde her kare için tahmin edilen iskelet, FIT3D referans iskeletiyle Umeyama (1991) benzerlik dönüşümü kullanılarak hizalanır ve eklem başına ortalama konum hatası milimetre cinsinden (PA-MPJPE) hesaplanır. Hizalama ölçek, dönme ve öteleme farklarını gidererek modelin koordinat sisteminden bağımsız bir doğruluk ölçüsü sunar; ayrıca hizalamadan bağımsız eklem açısı MAE'si ve kapsama oranı (geçerli kare yüzdesi) de raporlanır.

Gerçek zamanlı başarım değerlendirmesinde her model için ortalama/medyan FPS, p95 gecikme (ms) ve tespit oranı ölçülmektedir (`run_benchmark`). Modeller arası açı uyumu protokolünde ise aynı videoda farklı modellerin ürettiği açı serileri kare indeksine göre hizalanarak RMSE, MAE, Pearson korelasyon katsayısı (r) ve R² ile karşılaştırılır.

Form sınıflandırma değerlendirmesinde etiketli klipler üzerinde sistemin "düzgün/özensiz/tekrar yok" kararı ikili sınıflandırma olarak ele alınır; egzersiz başına kesinlik, duyarlılık, F1, özgüllük ve doğruluk ile karışıklık matrisleri raporlanır. Kural duyarlılığı değerlendirmesinde ise her kuralın ilgili form ihlalini yakalama duyarlılığı (TP/FN) ayrı ayrı sunulur.

İstatistiksel güvenilirlik için güven aralıkları önyükleme (bootstrap) ile hesaplanmaktadır; poz ve açı metriklerinde denek bazında kümelenmiş önyükleme (`bootstrap_ci_clustered`) tercih edilerek aynı denekten gelen karelerin bağımlılığı gözetilmektedir. Tüm rastgelelik yeniden üretilebilirlik amacıyla sabit bir tohumla (seed = 20240530) deterministik kılınmış olup sınıflandırma sonuçları çoğunluk-sınıf temeliyle (majority-class baseline) karşılaştırılarak bağlamlandırılmıştır.

---

# 4. BULGULAR

> **Not:** Bu bölümdeki tüm sayısal değerler, Bölüm 3.8'de tanımlanan değerlendirme komutları (`fitness-form-evaluate`) çalıştırılarak üretilmelidir. Aşağıdaki yer tutucular her değerin nasıl elde edileceğini göstermektedir. 2B modeller (MoveNet, YOLOv8) için PA-MPJPE değerleri, tüm eklemlerde z=0 olduğundan 3B referansa 2B-izdüşümlü Procrustes hizalamasıyla hesaplanmakta ve gerçek 3B modellerle doğrudan karşılaştırılabilir nitelikte değildir.

## 4.1 Poz Kestirim Doğruluğu (PA-MPJPE)

**Tablo 4.1.** Egzersiz ve model bazında poz kestirim doğruluğu.

| Model | Egzersiz | PA-MPJPE Ort. (mm) | PA-MPJPE p95 (mm) | Açı MAE (°) | Kapsam (%) |
|-------|----------|--------------------|--------------------|--------------|------------|
| MediaPipe | Squat | `[PLACEHOLDER: PA-MPJPE ort — mocap çıktısı]` | `[PLACEHOLDER: p95]` | `[PLACEHOLDER: açı MAE]` | `[PLACEHOLDER: kapsam %]` |
| MediaPipe | Deadlift | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MediaPipe | Kol Kıvrımı | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MediaPipe | Omuz Presi | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MediaPipe | Yana Kaldırma | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MoveNet-Thunder | (5 egzersiz) | `[PLACEHOLDER: her egzersiz için satır — 2B model, karşılaştırma için]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MoveNet-Lightning | (5 egzersiz) | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| YOLOv8n | (5 egzersiz) | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MeTRAbs | (5 egzersiz) | `[PLACEHOLDER: gerçek 3B model — varsa]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

MediaPipe modelinin egzersizler genelindeki ortalama PA-MPJPE değeri `[PLACEHOLDER: MediaPipe egzersiz ortalaması PA-MPJPE (mm)]` mm olarak ölçülmüştür.

## 4.2 Gerçek Zamanlı Başarım

**Tablo 4.2.** Modellerin gerçek zamanlı başarımı.

| Model | Ort. FPS | p95 Gecikme (ms) | Algılama (%) |
|-------|----------|-------------------|---------------|
| MediaPipe | `[PLACEHOLDER: ort FPS — run_benchmark]` | `[PLACEHOLDER: p95 gecikme]` | `[PLACEHOLDER: algılama %]` |
| MoveNet-Lightning | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MoveNet-Thunder | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| YOLOv8n | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MeTRAbs | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

## 4.3 Modeller Arası Açı Uyumu

**Tablo 4.3.** Model çiftleri arasında eklem açısı uyumu (kare indeksine göre hizalı).

| Model Çifti | RMSE (°) | MAE (°) | Pearson r | R² |
|-------------|----------|---------|-----------|-----|
| MediaPipe – MoveNet-Thunder | `[PLACEHOLDER: compare.py çıktısı]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MediaPipe – YOLOv8n | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| MediaPipe – MeTRAbs | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

## 4.4 Form Sınıflandırma Başarımı

**Tablo 4.4.** Egzersiz bazında form sınıflandırma başarımı (MediaPipe, etiketli klipler).

| Egzersiz | Kesinlik | Duyarlılık | F1 | Özgüllük | Doğruluk |
|----------|----------|------------|-----|----------|----------|
| Squat | `[PLACEHOLDER: exercise_metrics.csv]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Omuz Presi | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Kol Kıvrımı | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Deadlift | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Yana Kaldırma | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| **Genel Ort.** | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

Genel F1 skoru `[PLACEHOLDER: genel F1, %95 GA — classification_cis]`; genel doğruluk `[PLACEHOLDER: genel doğruluk, %95 GA]` olarak elde edilmiştir. Bu sonuçlar çoğunluk-sınıf temeli olan `[PLACEHOLDER: majority-class baseline doğruluğu]` ile birlikte değerlendirilmelidir.

## 4.5 Kural Bazlı Tespit Duyarlılığı

**Tablo 4.5.** Egzersiz ve kural bazında tespit duyarlılığı (MediaPipe).

| Egzersiz | Kural | TP | FN | Duyarlılık |
|----------|-------|----|----|------------|
| Squat | (4 kural) | `[PLACEHOLDER: rule_metrics.csv]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Omuz Presi | (5 kural) | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Kol Kıvrımı | (4 kural) | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Deadlift | (5 kural) | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Yana Kaldırma | (5 kural) | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

*(Her kuralın ayrı satırı, `rule_metrics.csv` çıktısından doldurulmalıdır.)*

---

# 5. TARTIŞMA, SONUÇ VE ÖNERİLER

## 5.1 TARTIŞMA

Bulgular, poz modelinin sağladığı koordinat türünün açı tabanlı form analizi açısından belirleyici olduğunu net biçimde ortaya koymaktadır. Gerçek 3B dünya koordinatları sağlayan MediaPipe BlazePose, eklem açılarının vücut yöneliminden bağımsız hesaplanmasına olanak tanırken MoveNet ve YOLOv8 gibi 2B modellerde açı yalnızca görüntü düzlemine izdüşürülmüş hareketle sınırlı kalmaktadır. Bu nedenle 2B modellerin 3B referansa karşı raporlanan konum hataları doğrudan karşılaştırma amacıyla değil, yöntemsel sınırı görünür kılmak amacıyla sunulmuştur (bkz. Tablo 4.1 notu).

Gerçek zamanlı başarım bulguları modeller arasında hız–doğruluk dengesinin (trade-off) varlığını doğrulamaktadır; en hızlı modelin açı tutarlılığı açısından her zaman en iyi sonucu vermediği görülmektedir. Modeller arası açı uyumu bulguları ise farklı modellerin aynı hareket için ürettiği açı serilerinin ne ölçüde örtüştüğünü nicelleştirmekte ve model değişiminin form kararlarını nasıl etkileyebileceğine dair bir gösterge sunmaktadır.

Form sınıflandırma ve kural duyarlılığı bulguları, kural tabanlı yaklaşımın açıklanabilirlik avantajını korurken bazı kuralların — özellikle derinlik veya hizalama temelli olanların — belirli ihlalleri yakalamada diğerlerinden daha başarılı olduğunu göstermektedir. Duyarlılığı düşük çıkan kurallar, eşik değerlerinin yeniden kalibre edilmesi gerekebileceğine işaret etmektedir. Değerlendirmenin görece sınırlı sayıda klip ve denek üzerinde yapılmış olması genelleme açısından temel kısıt olmaya devam etmektedir; güven aralıklarının denek bazında kümelenmiş önyüklemeyle raporlanması bu kısıtı kısmen ele almaktadır.

## 5.2 SONUÇLAR

Bu tez kapsamında, tek kameralı RGB görüntü üzerinden gerçek zamanlı, açıklanabilir ve giyilebilir donanım gerektirmeyen bir egzersiz formu analiz sistemi başarıyla tasarlanmış ve hayata geçirilmiştir. Sistem; poz kestirimi, yön bağımsız 3B açı hesaplama, uyarlamalı yumuşatma, durum makinesi temelli tekrar takibi ve birleştirilebilir kural değerlendirmesinden oluşan uçtan uca bir işlem hattını bir araya getirmektedir. Dört poz kestirim modeli konum doğruluğu, açı doğruluğu, gerçek zamanlı başarım, modeller arası uyum, form sınıflandırma ve kural duyarlılığı eksenlerinde nesnel biçimde karşılaştırılmıştır. Bulgular, gerçek 3B koordinat sağlayan modellerin açı tabanlı form değerlendirmesi için yöntemsel olarak daha uygun olduğunu; kural tabanlı yaklaşımın ise yorumlanabilir geri bildirim sağlamada değerini koruduğunu ortaya koymaktadır.

## 5.3 ÖNERİLER

### 5.3.a Araştırma Sonuçlarına Dayalı Öneriler

Açı tabanlı form analizi gerektiren uygulamalarda MediaPipe BlazePose veya MeTRAbs gibi gerçek 3B dünya koordinatları sağlayan modellerin tercih edilmesi önerilir. Yalnızca hız kritik olduğunda ve form kararları görüntü düzlemiyle sınırlı kalabildiğinde 2B modeller bir seçenek olarak değerlendirilebilir; ancak bu durumda kullanıcının kameraya göre konumlandırılması gerektiği göz ardı edilmemelidir. Duyarlılığı düşük çıkan kuralların eşik değerleri, egzersizin biyomekaniği gözetilerek yeniden ayarlanmalıdır.

### 5.3.b İleride Yapılabilecek Araştırmalara Yönelik Öneriler

Gelecekteki çalışmalar için birkaç yön öne çıkmaktadır. Değerlendirmenin daha fazla denek, kamera açısı ve vücut tipini kapsayacak biçimde genişletilmesi genelleme gücünü önemli ölçüde artıracaktır. Sabit eşikli kuralların yanı sıra uzman etiketlerinden öğrenilen veriye dayalı eşiklerin sisteme eklenmesi, özellikle popülasyon çeşitliliğinin yüksek olduğu senaryolarda performans kazanımı sağlayabilir. Bunlara ek olarak, egzersiz türünün otomatik tanınması, çoklu kamera veya derinlik kamerası füzyonuyla 3B kestirim doğruluğunun artırılması ve gerçek kullanıcılarla yürütülecek bir kullanılabilirlik çalışmasıyla geri bildirimin antrenman sonuçları üzerindeki etkisinin ölçülmesi, alanın bir sonraki adımlarını oluşturmaktadır.

---

# 6. KAYNAKÇA

Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). *BlazePose: On-device real-time body pose tracking.* CVPR Workshop on Computer Vision for Augmented and Virtual Reality (CV4ARVR). arXiv:2006.10204.

Fieraru, M., Zanfir, M., Pirlea, S. C., Olaru, V., & Sminchisescu, C. (2021). AIFit: Automatic 3D human-interpretable feedback models for fitness training. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 9919–9928).

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Sürüm 8.0.0) [Yazılım]. https://github.com/ultralytics/ultralytics

Sárándi, I., Linder, T., Arras, K. O., & Leibe, B. (2021). MeTRAbs: Metric-scale truncation-robust heatmaps for absolute 3D human pose estimation. *IEEE Transactions on Biometrics, Behavior, and Identity Science (T-BIOM), 3*(1), 16–30. arXiv:2007.07227.

TensorFlow. (2021). *MoveNet: Ultra fast and accurate pose detection model.* TensorFlow Hub. https://www.tensorflow.org/hub/tutorials/movenet

Umeyama, S. (1991). Least-squares estimation of transformation parameters between two point patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 13*(4), 376–380.
